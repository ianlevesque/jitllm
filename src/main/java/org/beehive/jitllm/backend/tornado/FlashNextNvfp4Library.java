package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import uk.ac.manchester.tornado.api.common.*;
import uk.ac.manchester.tornado.api.types.arrays.*;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;
import uk.ac.manchester.tornado.runtime.library.spi.*;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.*;

/** CUDA 13 cuBLASLt NVFP4 binding. Inputs remain packed E2M1 with tiled E4M3
 * block scales; outputs cross the same BF16 projection boundary as routed experts.
 * Numeric enum values/ABI are from CUDA 13 cublasLt.h and library_types.h.
 */
public final class FlashNextNvfp4Library implements TornadoLibraryProvider {
    public static final String NAME="jitllm/nvfp4";
    public static final long WORKSPACE_BYTES=1L<<20;
    private static final AtomicLong CONTEXTS=new AtomicLong(),PLANS=new AtomicLong(),CALLS=new AtomicLong();
    public record Statistics(long contexts,long plans,long workspaceBytes,long calls) {}
    public static Statistics statistics() {return new Statistics(CONTEXTS.get(),PLANS.get(),CONTEXTS.get()*WORKSPACE_BYTES,CALLS.get());}
    public static LibraryTaskDescriptor gemm(int rows,int batch,int columns,ByteArray weights,ByteArray weightScales,
                                            ByteArray input,ByteArray inputScales,FloatArray alphaBeta,BFloat16Array output) {
        if(rows<=0 || batch<=0 || columns<=0 || rows%16!=0 || columns%64!=0 || alphaBeta.getSize()<2
                || weights.getSize()!=Math.multiplyExact(rows,columns)/2 || input.getSize()!=Math.multiplyExact(batch,columns)/2
                || output.getSize()!=Math.multiplyExact(rows,batch)) throw new IllegalArgumentException("Invalid NVFP4 GEMM geometry");
        Access[] access=new Access[9];Arrays.fill(access,Access.READ_ONLY);access[8]=Access.WRITE_ONLY;
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("gemm")
                .withParameters(new Object[]{rows,batch,columns,weights,weightScales,input,inputScales,alphaBeta,output}).withAccess(access);
    }
    private static final class Native {
        static final SymbolLookup LT=required("libcublasLt.so.13","libcublasLt.so");
        static final SymbolLookup CUDA=required("libcudart.so.13","libcudart.so");
        static SymbolLookup required(String... names) {var lib=FFMSupport.loadLibrary(names);if(lib==null) throw new IllegalStateException("CUDA 13 library missing: "+Arrays.toString(names));return lib;}
        static MethodHandle call(SymbolLookup lib,String name,ValueLayout result,MemoryLayout... args) {
            var handle=FFMSupport.downcall(lib,FunctionDescriptor.of(result,args),name);
            if(handle==null) throw new IllegalStateException("CUDA symbol missing: "+name);return handle;
        }
        static final MethodHandle CREATE=call(LT,"cublasLtCreate",C_INT,C_POINTER),DESTROY=call(LT,"cublasLtDestroy",C_INT,C_LONG);
        static final MethodHandle DESC=call(LT,"cublasLtMatmulDescCreate",C_INT,C_POINTER,C_INT,C_INT),FREE_DESC=call(LT,"cublasLtMatmulDescDestroy",C_INT,C_LONG);
        static final MethodHandle ATTR=call(LT,"cublasLtMatmulDescSetAttribute",C_INT,C_LONG,C_INT,C_POINTER,C_LONG);
        static final MethodHandle LAYOUT=call(LT,"cublasLtMatrixLayoutCreate",C_INT,C_POINTER,C_INT,C_LONG,C_LONG,C_LONG),FREE_LAYOUT=call(LT,"cublasLtMatrixLayoutDestroy",C_INT,C_LONG);
        static final MethodHandle PREF=call(LT,"cublasLtMatmulPreferenceCreate",C_INT,C_POINTER),FREE_PREF=call(LT,"cublasLtMatmulPreferenceDestroy",C_INT,C_LONG);
        static final MethodHandle PREF_ATTR=call(LT,"cublasLtMatmulPreferenceSetAttribute",C_INT,C_LONG,C_INT,C_POINTER,C_LONG);
        static final MethodHandle HEURISTIC=call(LT,"cublasLtMatmulAlgoGetHeuristic",C_INT,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_INT,C_POINTER,C_POINTER);
        static final MethodHandle MATMUL=call(LT,"cublasLtMatmul",C_INT,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_LONG,C_POINTER,C_LONG,C_LONG,C_LONG);
        static final MethodHandle MALLOC=call(CUDA,"cudaMalloc",C_INT,C_POINTER,C_LONG),FREE=call(CUDA,"cudaFree",C_INT,C_LONG);
    }
    private static void check(int status,String action) {if(status!=0) throw new IllegalStateException(action+" failed with CUDA/cuBLAS status "+status);}
    private static RuntimeException failure(Throwable failure) {if(failure instanceof RuntimeException r) return r;if(failure instanceof Error e) throw e;return new IllegalStateException(failure);}
    private static long address(MemorySegment slot) {return slot.get(C_LONG,0);}
    private static final class Context implements LibraryContext {
        long handle,workspace;
        final Map<Shape,Plan> plans=new HashMap<>();
    }
    private record Shape(int rows,int batch,int columns) {}
    private static final class Plan implements AutoCloseable {
        final Arena arena=Arena.ofShared();long desc,a,b,c;
        final MemorySegment algorithm=arena.allocate(96,8),attribute=arena.allocate(8,8);
        boolean counted;
        void integer(int attr,int value) throws Throwable {attribute.set(C_INT,0,value);check((int)Native.ATTR.invokeExact(desc,attr,attribute,4L),"NVFP4 descriptor attribute "+attr);}
        void pointer(int attr,long value) throws Throwable {attribute.set(C_LONG,0,value);check((int)Native.ATTR.invokeExact(desc,attr,attribute,8L),"NVFP4 descriptor pointer "+attr);}
        long layout(int type,long rows,long columns,long stride) throws Throwable {
            var out=arena.allocate(8,8);check((int)Native.LAYOUT.invokeExact(out,type,rows,columns,stride),"NVFP4 matrix layout");return address(out);
        }
        Plan(Context context,Shape shape) {
            long preference=0;
            try {
                var slot=arena.allocate(8,8);check((int)Native.DESC.invokeExact(slot,68,0),"NVFP4 matmul descriptor");desc=address(slot);
                integer(3,1);integer(4,0);integer(2,1); // TN and device alpha/beta.
                integer(31,1);integer(32,1); // VEC16_UE4M3 for A/B.
                // The heuristic validates non-null scale pointers. prepare precedes
                // argument resolution; dispatch binds the actual scale buffers.
                pointer(17,context.workspace);pointer(18,context.workspace);
                a=layout(33,shape.columns,shape.rows,shape.columns);
                b=layout(33,shape.columns,shape.batch,shape.columns);
                c=layout(14,shape.rows,shape.batch,shape.rows);
                check((int)Native.PREF.invokeExact(slot),"NVFP4 heuristic preference");preference=address(slot);
                attribute.set(C_LONG,0,WORKSPACE_BYTES);check((int)Native.PREF_ATTR.invokeExact(preference,1,attribute,8L),"NVFP4 workspace preference");
                for(int attr=5;attr<=8;attr++) {attribute.set(C_INT,0,16);check((int)Native.PREF_ATTR.invokeExact(preference,attr,attribute,4L),"NVFP4 pointer alignment");}
                var count=arena.allocate(4,4);
                check((int)Native.HEURISTIC.invokeExact(context.handle,desc,a,b,c,c,preference,1,algorithm,count),"NVFP4 algorithm heuristic");
                if(count.get(C_INT,0)!=1 || algorithm.get(C_INT,72)!=0 || algorithm.get(C_LONG,64)>WORKSPACE_BYTES)
                    throw new IllegalStateException("No cuBLASLt NVFP4 algorithm for "+shape+" within "+WORKSPACE_BYTES+" bytes");
                counted=true;PLANS.incrementAndGet();
            } catch(Throwable error) {try {close();} catch(Exception cleanup) {error.addSuppressed(cleanup);}throw failure(error);}
            finally {if(preference!=0) try {check((int)Native.FREE_PREF.invokeExact(preference),"NVFP4 preference close");} catch(Throwable error) {throw failure(error);}}
        }
        @Override public void close() {
            try {
                for(long layout:new long[]{a,b,c}) if(layout!=0) check((int)Native.FREE_LAYOUT.invokeExact(layout),"NVFP4 layout close");
                a=b=c=0;if(desc!=0) {check((int)Native.FREE_DESC.invokeExact(desc),"NVFP4 descriptor close");desc=0;}
            } catch(Throwable error) {throw failure(error);}
            finally {if(counted) {counted=false;PLANS.decrementAndGet();}arena.close();}
        }
    }
    @Override public String libraryName() {return NAME;}
    @Override public boolean canHandle(TornadoXPUDevice device) {return device instanceof TornadoNativeStreamSupport;}
    @Override public LibraryContext createContext(TornadoXPUDevice device,long id) {
        var context=new Context();
        try(var arena=Arena.ofConfined()) {
            var slot=arena.allocate(8,8);check((int)Native.CREATE.invokeExact(slot),"NVFP4 cuBLASLt handle");context.handle=address(slot);
            check((int)Native.MALLOC.invokeExact(slot,WORKSPACE_BYTES),"NVFP4 device workspace");context.workspace=address(slot);
            CONTEXTS.incrementAndGet();return context;
        } catch(Throwable error) {
            if(context.handle!=0) try {check((int)Native.DESTROY.invokeExact(context.handle),"NVFP4 handle cleanup");} catch(Throwable cleanup) {error.addSuppressed(cleanup);}
            throw failure(error);
        }
    }
    private static Shape shape(Object[] args) {return new Shape((Integer)args[0],(Integer)args[1],(Integer)args[2]);}
    @Override public void prepare(LibraryTaskDescriptor descriptor,LibraryContext nativeContext) {
        var context=(Context)nativeContext;var shape=shape(descriptor.getParameters());
        context.plans.computeIfAbsent(shape,key->new Plan(context,key));
    }
    @Override public void dispatch(String function,LibraryInvocation invocation) {
        if(!"gemm".equals(function)) throw new IllegalArgumentException("Unsupported NVFP4 function "+function);
        var context=(Context)invocation.getContext();var shape=new Shape((Integer)invocation.getArg(0),(Integer)invocation.getArg(1),(Integer)invocation.getArg(2));
        var plan=context.plans.get(shape);if(plan==null) throw new IllegalStateException("NVFP4 plan was not prepared before capture");
        try {
            plan.pointer(17,invocation.getDevicePointer(4));plan.pointer(18,invocation.getDevicePointer(6));
            long alpha=invocation.getDevicePointer(7),beta=alpha+4,output=invocation.getDevicePointer(8);
            long stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
            check((int)Native.MATMUL.invokeExact(context.handle,plan.desc,alpha,invocation.getDevicePointer(3),plan.a,invocation.getDevicePointer(5),plan.b,beta,
                    output,plan.c,output,plan.c,plan.algorithm,context.workspace,WORKSPACE_BYTES,stream),"NVFP4 GEMM");
            CALLS.incrementAndGet();
        } catch(Throwable error) {throw failure(error);}
    }
    @Override public void destroyContext(LibraryContext nativeContext) {
        var context=(Context)nativeContext;
        try {
            for(var plan:context.plans.values()) plan.close();context.plans.clear();
            check((int)Native.FREE.invokeExact(context.workspace),"NVFP4 workspace close");context.workspace=0;
            check((int)Native.DESTROY.invokeExact(context.handle),"NVFP4 handle close");context.handle=0;CONTEXTS.decrementAndGet();
        } catch(Throwable error) {throw failure(error);}
    }
}
