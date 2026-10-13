package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.atomic.AtomicLong;
import uk.ac.manchester.tornado.api.common.*;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;
import uk.ac.manchester.tornado.runtime.library.spi.*;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.*;

/** SM120 CuTe-compatible Gemma normalization with an explicit CUDA approximate rsqrt.
 * Float buffers contain BF16-valued inputs and weights; output is rounded to BF16.
 */
public final class FlashNextRmsNormLibrary implements TornadoLibraryProvider {
    public static final String NAME="jitllm/gemma-rmsnorm";
    private static final AtomicLong CONTEXTS=new AtomicLong(),CALLS=new AtomicLong();
    public record Statistics(long contexts,long calls) {}
    public static Statistics statistics() {return new Statistics(CONTEXTS.get(),CALLS.get());}
    public static LibraryTaskDescriptor normalize(int rows,int width,float epsilon,FloatArray input,FloatArray weights,FloatArray output) {
        if(rows<=0 || width<=0 || width%8!=0 || !Float.isFinite(epsilon) || epsilon<=0 || weights.getSize()!=width
                || input.getSize()!=Math.multiplyExact(rows,width) || output.getSize()!=input.getSize()) throw new IllegalArgumentException("Invalid Gemma norm buffers");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("normalize")
                .withParameters(new Object[]{rows,width,epsilon,input,weights,output})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.WRITE_ONLY});
    }
    public static LibraryTaskDescriptor grouped(int groups,int width,float epsilon,FloatArray input,FloatArray weights,FloatArray output) {
        if(groups<=0 || width<=0 || width%512!=0 || width>16384 || !Float.isFinite(epsilon) || epsilon<=0
                || input.getSize()!=Math.multiplyExact(groups,width) || weights.getSize()!=input.getSize() || output.getSize()!=input.getSize())
            throw new IllegalArgumentException("Invalid grouped Gemma norm buffers");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("grouped")
                .withParameters(new Object[]{groups,width,epsilon,input,weights,output})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.WRITE_ONLY});
    }
    // Reduction geometry follows FlashInfer norm/kernels/rmsnorm.py (Apache-2.0).
    private static final String SOURCE="""
        extern "C" __global__ void gemma_norm(const float* input,const float* weights,float* output,int width,float epsilon) {
          int lane=threadIdx.x,threads=blockDim.x,row=blockIdx.x,warps=threads/32;
          __shared__ float sums[32];
          float sum=0.0f;
          for(int start=lane*8;start<width;start+=threads*8) {
            #pragma unroll
            for(int j=0;j<8;j++) {float x=input[row*width+start+j];sum+=x*x;}
          }
          for(int mask=1;mask<32;mask*=2) sum+=__shfl_xor_sync(0xffffffff,sum,mask);
          if(warps>1) {
            if(lane%32==0) sums[lane/32]=sum;
            __syncthreads();
            sum=lane%32<warps ? sums[lane%32] : 0.0f;
            for(int mask=1;mask<32;mask*=2) sum+=__shfl_xor_sync(0xffffffff,sum,mask);
          }
          float variance=sum/float(width)+epsilon,inverse;
          asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(variance));
          for(int start=lane*8;start<width;start+=threads*8) {
            #pragma unroll
            for(int j=0;j<8;j++) {
              float y=input[row*width+start+j]*inverse*(1.0f+weights[start+j]);
              unsigned int bits=__float_as_uint(y);
              bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;
              output[row*width+start+j]=__uint_as_float(bits);
            }
          }
        }
        extern "C" __global__ void grouped_gemma_norm(const float* input,const float* weights,float* output,int width,float epsilon) {
          int lane=threadIdx.x,start=blockIdx.x*width,warp=lane/32,warps=blockDim.x/32;
          __shared__ float sums[32];float sum=0.0f;
          #pragma unroll
          for(int pair=0;pair<8;pair++) {
            float x=input[start+lane*16+pair*2],y=input[start+lane*16+pair*2+1];
            sum+=x*x+y*y;
          }
          #pragma unroll
          for(int mask=16;mask>0;mask/=2) sum+=__shfl_xor_sync(0xffffffff,sum,mask);
          sums[warp]=sum;__syncthreads();
          if(warp==0) {
            sum=lane<warps ? sums[lane] : 0.0f;
            #pragma unroll
            for(int mask=16;mask>0;mask/=2) sum+=__shfl_xor_sync(0xffffffff,sum,mask);
            sums[lane]=rsqrtf(sum/float(width)+epsilon);
          }
          __syncthreads();float inverse=sums[warp];
          #pragma unroll
          for(int j=0;j<16;j++) {
            int at=start+lane*16+j;float y=input[at]*inverse*(1.0f+weights[at]);
            unsigned int bits=__float_as_uint(y);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;
            output[at]=__uint_as_float(bits);
          }
        }
        """;
    private static final class Native {
        static final SymbolLookup CUDA=required("libcuda.so.1","libcuda.so"),RTC=required("libnvrtc.so.13","libnvrtc.so");
        static SymbolLookup required(String... names) {var lookup=FFMSupport.loadLibrary(names);if(lookup==null) throw new IllegalStateException("Missing CUDA/NVRTC library");return lookup;}
        static MethodHandle call(SymbolLookup library,String name,MemoryLayout... args) {var handle=FFMSupport.downcall(library,FunctionDescriptor.of(C_INT,args),name);if(handle==null) throw new IllegalStateException("Missing CUDA symbol "+name);return handle;}
        static final MethodHandle CREATE=call(RTC,"nvrtcCreateProgram",C_POINTER,C_POINTER,C_POINTER,C_INT,C_POINTER,C_POINTER),COMPILE=call(RTC,"nvrtcCompileProgram",C_LONG,C_INT,C_POINTER),
                SIZE=call(RTC,"nvrtcGetPTXSize",C_LONG,C_POINTER),PTX=call(RTC,"nvrtcGetPTX",C_LONG,C_POINTER),DESTROY=call(RTC,"nvrtcDestroyProgram",C_POINTER),
                LOGSIZE=call(RTC,"nvrtcGetProgramLogSize",C_LONG,C_POINTER),LOG=call(RTC,"nvrtcGetProgramLog",C_LONG,C_POINTER);
        static final MethodHandle DEVICE=call(CUDA,"cuCtxGetDevice",C_POINTER),ATTRIBUTE=call(CUDA,"cuDeviceGetAttribute",C_POINTER,C_INT,C_INT),
                LOAD=call(CUDA,"cuModuleLoadData",C_POINTER,C_POINTER),FUNCTION=call(CUDA,"cuModuleGetFunction",C_POINTER,C_LONG,C_POINTER),UNLOAD=call(CUDA,"cuModuleUnload",C_LONG),
                LAUNCH=call(CUDA,"cuLaunchKernel",C_LONG,C_INT,C_INT,C_INT,C_INT,C_INT,C_INT,C_INT,C_LONG,C_POINTER,C_POINTER);
    }
    private static void check(int status,String action) {if(status!=0) throw new IllegalStateException(action+" failed with CUDA/NVRTC status "+status);}
    private static RuntimeException failure(Throwable error) {if(error instanceof RuntimeException r) return r;if(error instanceof Error e) throw e;return new IllegalStateException(error);}
    private static final class Context implements LibraryContext {
        long module,function,groupFunction;
        void prepare() {
            if(module!=0) return;
            long program=0;
            try(var arena=Arena.ofConfined()) {
                var slot=arena.allocate(8,8);check((int)Native.DEVICE.invokeExact(slot),"CUDA norm device");int device=slot.get(C_INT,0);
                check((int)Native.ATTRIBUTE.invokeExact(slot,75,device),"CUDA compute major");int major=slot.get(C_INT,0);
                check((int)Native.ATTRIBUTE.invokeExact(slot,76,device),"CUDA compute minor");int minor=slot.get(C_INT,0);
                check((int)Native.CREATE.invokeExact(slot,arena.allocateFrom(SOURCE),arena.allocateFrom("jitllm_gemma_norm.cu"),0,MemorySegment.NULL,MemorySegment.NULL),"NVRTC norm program");program=slot.get(C_LONG,0);
                var options=arena.allocate(16,8);options.set(C_POINTER,0,arena.allocateFrom("--gpu-architecture=compute_"+major+minor));options.set(C_POINTER,8,arena.allocateFrom("--fmad=false"));
                int status=(int)Native.COMPILE.invokeExact(program,2,options);
                if(status!=0) {
                    check((int)Native.LOGSIZE.invokeExact(program,slot),"NVRTC log size");var log=arena.allocate(slot.get(C_LONG,0),1);
                    check((int)Native.LOG.invokeExact(program,log),"NVRTC log");throw new IllegalStateException("NVRTC norm compile failed: "+log.getString(0));
                }
                check((int)Native.SIZE.invokeExact(program,slot),"NVRTC PTX size");var code=arena.allocate(slot.get(C_LONG,0),1);check((int)Native.PTX.invokeExact(program,code),"NVRTC PTX");
                check((int)Native.LOAD.invokeExact(slot,code),"CUDA norm module");module=slot.get(C_LONG,0);
                check((int)Native.FUNCTION.invokeExact(slot,module,arena.allocateFrom("gemma_norm")),"CUDA norm function");function=slot.get(C_LONG,0);
                check((int)Native.FUNCTION.invokeExact(slot,module,arena.allocateFrom("grouped_gemma_norm")),"CUDA grouped norm function");groupFunction=slot.get(C_LONG,0);CONTEXTS.incrementAndGet();
            } catch(Throwable error) {
                if(module!=0) try {check((int)Native.UNLOAD.invokeExact(module),"CUDA norm cleanup");module=0;} catch(Throwable closing) {error.addSuppressed(closing);}
                throw failure(error);
            } finally {
                if(program!=0) try(var arena=Arena.ofConfined()) {var slot=arena.allocate(8,8);slot.set(C_LONG,0,program);check((int)Native.DESTROY.invokeExact(slot),"NVRTC norm program close");}
                catch(Throwable error) {throw failure(error);}
            }
        }
    }
    @Override public String libraryName() {return NAME;}
    @Override public boolean canHandle(TornadoXPUDevice device) {return device instanceof TornadoNativeStreamSupport;}
    @Override public LibraryContext createContext(TornadoXPUDevice device,long id) {return new Context();}
    @Override public void prepare(LibraryTaskDescriptor descriptor,LibraryContext context) {if(!descriptor.getFunctionName().equals("normalize") && !descriptor.getFunctionName().equals("grouped")) throw new IllegalArgumentException("Unknown norm function");((Context)context).prepare();}
    @Override public void dispatch(String function,LibraryInvocation invocation) {
        var context=(Context)invocation.getContext();int rows=(Integer)invocation.getArg(0),width=(Integer)invocation.getArg(1);
        boolean grouped=function.equals("grouped");int threads=grouped ? width/16 : width<=3072 ? 32 : width<=6144 ? 64 : width<=16384 ? 128 : 256;
        long kernel=grouped ? context.groupFunction : context.function;long stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try(var arena=Arena.ofConfined()) {
            var parameters=arena.allocate(40,8);
            for(int i=0;i<3;i++) {var argument=arena.allocate(8,8);argument.set(C_LONG,0,invocation.getDevicePointer(i+3));parameters.set(C_POINTER,i*8L,argument);}
            var d=arena.allocate(4,4);d.set(C_INT,0,width);parameters.set(C_POINTER,24,d);
            var e=arena.allocate(4,4);e.set(C_FLOAT,0,(Float)invocation.getArg(2));parameters.set(C_POINTER,32,e);
            check((int)Native.LAUNCH.invokeExact(kernel,rows,1,1,threads,1,1,0,stream,parameters,MemorySegment.NULL),"CUDA Gemma norm launch");CALLS.incrementAndGet();
        } catch(Throwable error) {throw failure(error);}
    }
    @Override public void destroyContext(LibraryContext context) {
        var nativeContext=(Context)context;if(nativeContext.module==0) return;
        try {check((int)Native.UNLOAD.invokeExact(nativeContext.module),"CUDA norm module close");nativeContext.module=nativeContext.function=0;CONTEXTS.decrementAndGet();}
        catch(Throwable error) {throw failure(error);}
    }
}
