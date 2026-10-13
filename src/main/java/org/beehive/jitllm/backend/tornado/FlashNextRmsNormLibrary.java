package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.atomic.AtomicLong;
import uk.ac.manchester.tornado.api.common.*;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
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
    public static LibraryTaskDescriptor combine(int branches,int width,FloatArray block,FloatArray residual,FloatArray normalized,FloatArray weights,FloatArray output) {
        int row=Math.multiplyExact(branches,width);
        if(branches<=0 || width<=0 || width%8!=0 || row%2048!=0 || block.getSize()!=width || residual.getSize()!=row
                || normalized.getSize()!=row || weights.getSize()!=Math.multiplyExact(branches,row) || output.getSize()!=row)
            throw new IllegalArgumentException("Invalid native HC combine buffers");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("combine")
                .withParameters(new Object[]{branches,width,block,residual,normalized,weights,output})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.WRITE_ONLY});
    }
    public static LibraryTaskDescriptor activate(int branches,int length,FloatArray low,IntArray fused) {
        if(branches<=0 || length<=0 || low.getSize()!=length || fused.getSize()!=1) throw new IllegalArgumentException("Invalid HC low activation buffers");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("activate")
                .withParameters(new Object[]{branches,length,low,fused})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_WRITE,Access.READ_ONLY});
    }
    public static LibraryTaskDescriptor mix(int branches,int width,FloatArray normalized,FloatArray gates,IntArray fused,FloatArray output) {
        if(branches!=4 || width<=0 || normalized.getSize()!=Math.multiplyExact(branches,width) || gates.getSize()!=normalized.getSize()
                || fused.getSize()!=1 || output.getSize()!=width) throw new IllegalArgumentException("Invalid HC mix buffers");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("mix")
                .withParameters(new Object[]{branches,width,normalized,gates,fused,output})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.WRITE_ONLY});
    }
    // Reduction geometry follows FlashInfer norm/kernels/rmsnorm.py (Apache-2.0).
    private static final String SOURCE="""
        __device__ float hc_bf(float value) {
          unsigned int bits=__float_as_uint(value);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;
          return __uint_as_float(bits);
        }
        // PTX operations used by SGLang's SM120 hc_mix_triton.py persistent kernel.
        __device__ float hc_sigmoid(float value) {
          float exponent=__fmul_rn(-value,1.4426950408889634f),power,result;
          asm volatile("ex2.approx.f32 %0, %1;" : "=f"(power) : "f"(exponent));
          float denominator=__fadd_rn(1.0f,power);
          asm volatile("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(1.0f),"f"(denominator));
          return result;
        }
        extern "C" __global__ void hc_activate(float* low,const int* fused,int length,int branches) {
          int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=length) return;
          float value=(*fused ? low[at] : hc_bf(low[at]))/float(branches);
          low[at]=hc_bf(*fused ? __fmul_rn(value,hc_sigmoid(value)) : float(double(value)/(1.0+exp(-double(value)))));
        }
        extern "C" __global__ void hc_mix(const float* normalized,const float* gates,const int* fused,float* output,int width,int branches) {
          int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=width) return;
          if(*fused) {
            float g0=hc_sigmoid(gates[at]),g1=hc_sigmoid(gates[width+at]);
            float g2=hc_sigmoid(gates[2*width+at]),g3=hc_sigmoid(gates[3*width+at]);
            float a=fmaf(g0,normalized[at],__fmul_rn(g2,normalized[2*width+at]));
            float b=fmaf(g1,normalized[width+at],__fmul_rn(g3,normalized[3*width+at]));
            output[at]=hc_bf(__fmul_rn(__fadd_rn(a,b),0.25f));
          } else {
            float sum=0.0f;
            for(int branch=0;branch<branches;branch++) {
              int j=branch*width+at;float gain=float(1.0/(1.0+exp(-double(hc_bf(gates[j])))));
              sum=fmaf(gain,normalized[j],sum);
            }
            output[at]=hc_bf(sum/float(branches));
          }
        }
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
        // The gate reduction follows SGLang hc_combine.cuh (Apache-2.0): eight
        // values per lane, paired products, descending XOR, then eight warp sums.
        extern "C" __global__ void hc_combine(const float* block,const float* residual,const float* normalized,const float* weights,float* output,int branches,int width) {
          int lane=threadIdx.x,branch=blockIdx.x,row=branches*width;
          __shared__ float sums[8];__shared__ float gain;float sum=0.0f;
          for(int start=lane*8;start<row;start+=2048) {
            #pragma unroll
            for(int pair=0;pair<4;pair++) {
              int at=start+pair*2;
              sum+=normalized[at]*weights[branch*row+at]+normalized[at+1]*weights[branch*row+at+1];
            }
          }
          #pragma unroll
          for(int mask=16;mask>0;mask/=2) sum+=__shfl_xor_sync(0xffffffff,sum,mask);
          if(lane%32==0) sums[lane/32]=sum;__syncthreads();
          if(lane==0) {
            float total=0.0f;
            #pragma unroll
            for(int warp=0;warp<8;warp++) total+=sums[warp];
            gain=2.0f/(1.0f+expf(-total/float(branches)));
          }
          __syncthreads();
          for(int column=lane;column<width;column+=256) {
            int at=branch*width+column;float value=fmaf(gain,block[column],residual[at]);
            unsigned int bits=__float_as_uint(value);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;
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
        long module,function,groupFunction,combineFunction,activateFunction,mixFunction;
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
                check((int)Native.FUNCTION.invokeExact(slot,module,arena.allocateFrom("grouped_gemma_norm")),"CUDA grouped norm function");groupFunction=slot.get(C_LONG,0);
                check((int)Native.FUNCTION.invokeExact(slot,module,arena.allocateFrom("hc_combine")),"CUDA HC combine function");combineFunction=slot.get(C_LONG,0);
                check((int)Native.FUNCTION.invokeExact(slot,module,arena.allocateFrom("hc_activate")),"CUDA HC activation function");activateFunction=slot.get(C_LONG,0);
                check((int)Native.FUNCTION.invokeExact(slot,module,arena.allocateFrom("hc_mix")),"CUDA HC mix function");mixFunction=slot.get(C_LONG,0);CONTEXTS.incrementAndGet();
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
    @Override public void prepare(LibraryTaskDescriptor descriptor,LibraryContext context) {
        if(!java.util.Set.of("normalize","grouped","combine","activate","mix").contains(descriptor.getFunctionName())) throw new IllegalArgumentException("Unknown norm function");
        ((Context)context).prepare();
    }
    @Override public void dispatch(String function,LibraryInvocation invocation) {
        if(function.equals("combine")) {dispatchCombine(invocation);return;}
        if(function.equals("activate") || function.equals("mix")) {dispatchMix(function,invocation);return;}
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
    private static void dispatchMix(String function,LibraryInvocation invocation) {
        var context=(Context)invocation.getContext();int branches=(Integer)invocation.getArg(0),width=(Integer)invocation.getArg(1);
        boolean activate=function.equals("activate");int pointers=activate ? 2 : 4;
        long stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try(var arena=Arena.ofConfined()) {
            var parameters=arena.allocate((pointers+2)*8L,8);
            for(int i=0;i<pointers;i++) {var argument=arena.allocate(8,8);argument.set(C_LONG,0,invocation.getDevicePointer(i+2));parameters.set(C_POINTER,i*8L,argument);}
            var w=arena.allocate(4,4);w.set(C_INT,0,width);parameters.set(C_POINTER,pointers*8L,w);
            var b=arena.allocate(4,4);b.set(C_INT,0,branches);parameters.set(C_POINTER,(pointers+1)*8L,b);
            check((int)Native.LAUNCH.invokeExact(activate ? context.activateFunction : context.mixFunction,(width+255)/256,1,1,256,1,1,0,stream,parameters,MemorySegment.NULL),"CUDA HC mix/activation launch");CALLS.incrementAndGet();
        } catch(Throwable error) {throw failure(error);}
    }
    private static void dispatchCombine(LibraryInvocation invocation) {
        var context=(Context)invocation.getContext();int branches=(Integer)invocation.getArg(0),width=(Integer)invocation.getArg(1);
        long stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try(var arena=Arena.ofConfined()) {
            var parameters=arena.allocate(56,8);
            for(int i=0;i<5;i++) {var argument=arena.allocate(8,8);argument.set(C_LONG,0,invocation.getDevicePointer(i+2));parameters.set(C_POINTER,i*8L,argument);}
            var b=arena.allocate(4,4);b.set(C_INT,0,branches);parameters.set(C_POINTER,40,b);
            var w=arena.allocate(4,4);w.set(C_INT,0,width);parameters.set(C_POINTER,48,w);
            check((int)Native.LAUNCH.invokeExact(context.combineFunction,branches,1,1,256,1,1,0,stream,parameters,MemorySegment.NULL),"CUDA HC combine launch");CALLS.incrementAndGet();
        } catch(Throwable error) {throw failure(error);}
    }
    @Override public void destroyContext(LibraryContext context) {
        var nativeContext=(Context)context;if(nativeContext.module==0) return;
        try {check((int)Native.UNLOAD.invokeExact(nativeContext.module),"CUDA norm module close");nativeContext.module=nativeContext.function=0;CONTEXTS.decrementAndGet();}
        catch(Throwable error) {throw failure(error);}
    }
}
