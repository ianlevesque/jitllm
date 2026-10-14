package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.concurrent.atomic.AtomicLong;
import uk.ac.manchester.tornado.api.common.*;
import uk.ac.manchester.tornado.api.types.arrays.*;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;
import uk.ac.manchester.tornado.runtime.library.spi.*;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.*;

/** Staged SM120 persistent HC mix, including its atomic FP32 down reduction.
 * The prepared cubin runs directly on the execution-plan stream; Python is not used at inference.
 */
public final class FlashNextHcMixLibrary implements TornadoLibraryProvider {
    public static final String NAME="jitllm/hc-mix";
    private static final int SHARED_BYTES=51200;
    private static final AtomicLong CONTEXTS=new AtomicLong(),CALLS=new AtomicLong();
    public record Statistics(long contexts,long calls) {}
    public static Statistics statistics() {return new Statistics(CONTEXTS.get(),CALLS.get());}
    public static LibraryTaskDescriptor mix(BFloat16Array input,BFloat16Array down,BFloat16Array up,
            FloatArray raw,BFloat16Array output,IntArray counters) {
        if(input.getSize()!=10240 || down.getSize()!=320*10240 || up.getSize()!=down.getSize()
                || raw.getSize()!=16*320 || output.getSize()!=2560 || counters.getSize()!=3)
            throw new IllegalArgumentException("Native HC mix requires the qualified single-row BF16 geometry");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("mix")
                .withParameters(new Object[]{input,down,up,raw,output,counters})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_WRITE,Access.WRITE_ONLY,Access.READ_WRITE});
    }
    private static final class Native {
        static final SymbolLookup CUDA=required();
        static SymbolLookup required() {var library=FFMSupport.loadLibrary("libcuda.so.1");if(library==null) throw new IllegalStateException("Missing HC CUDA driver");return library;}
        static MethodHandle call(String name,MemoryLayout... args) {
            var handle=FFMSupport.downcall(CUDA,FunctionDescriptor.of(C_INT,args),name);
            if(handle==null) throw new IllegalStateException("Missing HC CUDA symbol "+name);return handle;
        }
        static final MethodHandle DEVICE=call("cuCtxGetDevice",C_POINTER),ATTRIBUTE=call("cuDeviceGetAttribute",C_POINTER,C_INT,C_INT),
                LOAD=call("cuModuleLoadData",C_POINTER,C_POINTER),FUNCTION=call("cuModuleGetFunction",C_POINTER,C_LONG,C_POINTER),
                FUNC_ATTRIBUTE=call("cuFuncSetAttribute",C_LONG,C_INT,C_INT),UNLOAD=call("cuModuleUnload",C_LONG),
                OCCUPANCY=call("cuOccupancyMaxActiveBlocksPerMultiprocessor",C_POINTER,C_LONG,C_INT,C_LONG),
                LAUNCH=call("cuLaunchKernel",C_LONG,C_INT,C_INT,C_INT,C_INT,C_INT,C_INT,C_INT,C_LONG,C_POINTER,C_POINTER);
    }
    private static void check(int status,String action) {if(status!=0) throw new IllegalStateException(action+" failed with CUDA status "+status);}
    private static RuntimeException failure(Throwable error) {if(error instanceof RuntimeException r) return r;if(error instanceof Error e) throw e;return new IllegalStateException(error);}
    private static final class Context implements LibraryContext {
        long module,function;int sms;boolean counted;
        void prepare() {
            if(counted) return;
            try(var arena=Arena.ofConfined()) {
                var slot=arena.allocate(8,8);check((int)Native.DEVICE.invokeExact(slot),"HC CUDA device");int device=slot.get(C_INT,0);
                check((int)Native.ATTRIBUTE.invokeExact(slot,75,device),"HC compute major");int major=slot.get(C_INT,0);
                check((int)Native.ATTRIBUTE.invokeExact(slot,76,device),"HC compute minor");int minor=slot.get(C_INT,0);
                if(major!=12 || minor!=0) throw new IllegalStateException("Native HC cubin requires SM120");
                check((int)Native.ATTRIBUTE.invokeExact(slot,16,device),"HC SM count");sms=slot.get(C_INT,0);
                if(sms<=0 || sms%2!=0) throw new IllegalStateException("HC cubin requires the compiled even-SM launch specialization");
                String file=System.getenv("JITLLM_HC_CUBIN"),digest=System.getenv("JITLLM_HC_SHA256");
                if(file==null || digest==null) throw new IllegalStateException("Run prepare_jitllm_hc.py and export the prepared HC environment");
                var path=Path.of(file);if(!path.isAbsolute() || !Files.isRegularFile(path)) throw new IllegalStateException("Missing HC cubin "+path);
                byte[] code=Files.readAllBytes(path);
                if(!digest.equals(HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(code)))) throw new IllegalStateException("Prepared HC cubin changed");
                check((int)Native.LOAD.invokeExact(slot,arena.allocateFrom(ValueLayout.JAVA_BYTE,code)),"HC CUDA module");module=slot.get(C_LONG,0);
                check((int)Native.FUNCTION.invokeExact(slot,module,arena.allocateFrom("_hc_mix_persistent_kernel")),"HC CUDA function");function=slot.get(C_LONG,0);
                check((int)Native.FUNC_ATTRIBUTE.invokeExact(function,8,SHARED_BYTES),"HC dynamic shared-memory limit");
                check((int)Native.OCCUPANCY.invokeExact(slot,function,256,(long)SHARED_BYTES),"HC resident CTA capacity");
                if(slot.get(C_INT,0)<1) throw new IllegalStateException("HC grid barrier requires one resident CTA per SM");
                counted=true;CONTEXTS.incrementAndGet();
            } catch(Throwable error) {try {close();} catch(Throwable closing) {error.addSuppressed(closing);}throw failure(error);}
        }
        void close() {
            if(module!=0) try {check((int)Native.UNLOAD.invokeExact(module),"HC CUDA module close");module=function=0;}
            catch(Throwable error) {throw failure(error);}
            finally {if(counted) {counted=false;CONTEXTS.decrementAndGet();}}
        }
    }
    @Override public String libraryName() {return NAME;}
    @Override public boolean canHandle(TornadoXPUDevice device) {return device instanceof TornadoNativeStreamSupport;}
    @Override public LibraryContext createContext(TornadoXPUDevice device,long id) {return new Context();}
    @Override public void prepare(LibraryTaskDescriptor descriptor,LibraryContext context) {
        if(!descriptor.getFunctionName().equals("mix")) throw new IllegalArgumentException("Unknown HC function");((Context)context).prepare();
    }
    @Override public void dispatch(String function,LibraryInvocation invocation) {
        var context=(Context)invocation.getContext();long stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try(var arena=Arena.ofConfined()) {
            var args=arena.allocate(13*8L,8);
            for(int i=0;i<6;i++) {var cell=arena.allocate(8,8);cell.set(C_LONG,0,invocation.getDevicePointer(i));args.set(C_POINTER,i*8L,cell);}
            int[] scalar={10240,320,2560,context.sms};
            for(int i=0;i<4;i++) {var cell=arena.allocate(4,4);cell.set(C_INT,0,scalar[i]);args.set(C_POINTER,(i+6)*8L,cell);}
            var inverse=arena.allocate(4,4);inverse.set(C_FLOAT,0,.25f);args.set(C_POINTER,80,inverse);
            for(int i=11;i<13;i++) {var cell=arena.allocate(8,8);cell.set(C_LONG,0,0);args.set(C_POINTER,i*8L,cell);}
            check((int)Native.LAUNCH.invokeExact(context.function,context.sms,1,1,256,1,1,SHARED_BYTES,stream,args,MemorySegment.NULL),"HC persistent launch");CALLS.incrementAndGet();
        } catch(Throwable error) {throw failure(error);}
    }
    @Override public void destroyContext(LibraryContext context) {((Context)context).close();}
}
