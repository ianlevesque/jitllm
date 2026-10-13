package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.atomic.AtomicLong;
import uk.ac.manchester.tornado.api.common.*;
import uk.ac.manchester.tornado.api.types.arrays.*;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;
import uk.ac.manchester.tornado.runtime.library.spi.*;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.*;

/** BF16 shared-expert GEMMs with an explicit workspace rebind on each caller's stream. */
public final class FlashNextSharedBf16Library implements TornadoLibraryProvider {
    public static final String NAME="jitllm/shared-bf16";
    public static final long WORKSPACE_BYTES=32L<<20;
    private static final AtomicLong CONTEXTS=new AtomicLong(),CALLS=new AtomicLong();
    public record Statistics(long contexts,long workspaceBytes,long calls) {}
    public static Statistics statistics() {return new Statistics(CONTEXTS.get(),CONTEXTS.get()*WORKSPACE_BYTES,CALLS.get());}
    public static LibraryTaskDescriptor project(int rows,int columns,ByteArray weights,BFloat16Array input,BFloat16Array output) {
        if(rows<=0 || columns<=0 || weights.getSize()!=Math.multiplyExact(2,Math.multiplyExact(rows,columns)) || input.getSize()!=columns || output.getSize()!=rows)
            throw new IllegalArgumentException("Invalid BF16 shared projection");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("project")
                .withParameters(new Object[]{rows,columns,weights,input,output})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.WRITE_ONLY});
    }
    private static final class Native {
        static final SymbolLookup BLAS=required("libcublas.so.13","libcublas.so"),CUDA=required("libcudart.so.13","libcudart.so");
        static SymbolLookup required(String... names) {var lib=FFMSupport.loadLibrary(names);if(lib==null) throw new IllegalStateException("Missing BF16 CUDA library");return lib;}
        static MethodHandle call(SymbolLookup lib,String name,MemoryLayout... args) {var h=FFMSupport.downcall(lib,FunctionDescriptor.of(C_INT,args),name);if(h==null) throw new IllegalStateException("Missing BF16 symbol "+name);return h;}
        static final MethodHandle CREATE=call(BLAS,"cublasCreate_v2",C_POINTER),DESTROY=call(BLAS,"cublasDestroy_v2",C_LONG),
                STREAM=call(BLAS,"cublasSetStream_v2",C_LONG,C_LONG),WORKSPACE=call(BLAS,"cublasSetWorkspace_v2",C_LONG,C_LONG,C_LONG),
                GEMM=call(BLAS,"cublasGemmEx",C_LONG,C_INT,C_INT,C_INT,C_INT,C_INT,C_POINTER,C_LONG,C_INT,C_INT,C_LONG,C_INT,C_INT,C_POINTER,C_LONG,C_INT,C_INT,C_INT,C_INT),
                MALLOC=call(CUDA,"cudaMalloc",C_POINTER,C_LONG),FREE=call(CUDA,"cudaFree",C_LONG);
    }
    private static void check(int status,String action) {if(status!=0) throw new IllegalStateException(action+" failed with cuBLAS/CUDA status "+status);}
    private static RuntimeException failure(Throwable error) {if(error instanceof RuntimeException r) return r;if(error instanceof Error e) throw e;return new IllegalStateException(error);}
    private static final class Context implements LibraryContext {
        long handle,workspace;boolean counted;
        void prepare() {
            if(counted) return;
            try(var arena=Arena.ofConfined()) {
                var slot=arena.allocate(8,8);check((int)Native.CREATE.invokeExact(slot),"Shared BF16 handle");handle=slot.get(C_LONG,0);
                check((int)Native.MALLOC.invokeExact(slot,WORKSPACE_BYTES),"Shared BF16 workspace");workspace=slot.get(C_LONG,0);
                counted=true;CONTEXTS.incrementAndGet();
            } catch(Throwable error) {try {close();} catch(Throwable cleanup) {error.addSuppressed(cleanup);}throw failure(error);}
        }
        void close() {
            Throwable error=null;
            try {if(handle!=0) {check((int)Native.DESTROY.invokeExact(handle),"Shared BF16 handle close");handle=0;}} catch(Throwable failure) {error=failure;}
            try {if(workspace!=0) {check((int)Native.FREE.invokeExact(workspace),"Shared BF16 workspace close");workspace=0;}} catch(Throwable failure) {if(error==null) error=failure;else error.addSuppressed(failure);}
            if(counted) {counted=false;CONTEXTS.decrementAndGet();}if(error!=null) throw failure(error);
        }
    }
    @Override public String libraryName() {return NAME;}
    @Override public boolean canHandle(TornadoXPUDevice device) {return device instanceof TornadoNativeStreamSupport;}
    @Override public LibraryContext createContext(TornadoXPUDevice device,long id) {return new Context();}
    @Override public void prepare(LibraryTaskDescriptor descriptor,LibraryContext context) {((Context)context).prepare();}
    @Override public void dispatch(String function,LibraryInvocation invocation) {
        var context=(Context)invocation.getContext();int rows=(Integer)invocation.getArg(0),columns=(Integer)invocation.getArg(1);
        long stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try(var arena=Arena.ofConfined()) {
            check((int)Native.STREAM.invokeExact(context.handle,stream),"Shared BF16 caller stream");
            check((int)Native.WORKSPACE.invokeExact(context.handle,context.workspace,WORKSPACE_BYTES),"Shared BF16 workspace rebind");
            var alpha=arena.allocate(4,4);alpha.set(C_FLOAT,0,1);var beta=arena.allocate(4,4);
            check((int)Native.GEMM.invokeExact(context.handle,1,0,rows,1,columns,alpha,invocation.getDevicePointer(2),14,columns,
                    invocation.getDevicePointer(3),14,columns,beta,invocation.getDevicePointer(4),14,rows,68,-1),"Shared BF16 GEMM");CALLS.incrementAndGet();
        } catch(Throwable error) {throw failure(error);}
    }
    @Override public void destroyContext(LibraryContext context) {((Context)context).close();}
}
