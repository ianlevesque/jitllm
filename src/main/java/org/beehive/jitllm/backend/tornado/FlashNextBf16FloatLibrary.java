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

/** Native BF16 tensor-core projections with FP32 output and optional K partitions. */
public final class FlashNextBf16FloatLibrary implements TornadoLibraryProvider {
    public static final String NAME="jitllm/bf16-float";
    public static final long WORKSPACE_BYTES=1L<<20;
    private static final AtomicLong CONTEXTS=new AtomicLong(),PLANS=new AtomicLong(),CALLS=new AtomicLong();
    public record Statistics(long contexts,long plans,long workspaceBytes,long calls) {}
    public static Statistics statistics() {return new Statistics(CONTEXTS.get(),PLANS.get(),CONTEXTS.get()*WORKSPACE_BYTES,CALLS.get());}
    public static LibraryTaskDescriptor partitions(int rows,int batch,int columns,BFloat16Array weights,
            BFloat16Array input,FloatArray alphaBeta,FloatArray partials,int partition) {
        if(rows<=0 || batch<=0 || columns<=0 || rows%16!=0 || batch%16!=0 || partition<=0 || partition%16!=0 || columns%partition!=0
                || alphaBeta.getSize()<2 || weights.getSize()!=Math.multiplyExact(rows,columns)
                || input.getSize()!=Math.multiplyExact(batch,columns)
                || partials.getSize()!=Math.multiplyExact(Math.multiplyExact(rows,batch),columns/partition))
            throw new IllegalArgumentException("Invalid block BF16/F32 GEMM geometry");
        Access[] access=new Access[8];Arrays.fill(access,Access.READ_ONLY);access[6]=Access.WRITE_ONLY;
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("partitions")
                .withParameters(new Object[]{rows,batch,columns,weights,input,alphaBeta,partials,partition}).withAccess(access);
    }
    public static LibraryTaskDescriptor gqaScores(int capacity,int dimension,int groups,IntArray control,BFloat16Array keys,BFloat16Array queries,FloatArray alphaBeta,FloatArray scores) {
        if(capacity<=0 || capacity%32!=0 || dimension<=0 || dimension%16!=0 || groups<=0 || control.getSize()<6 || alphaBeta.getSize()<2
                || keys.getSize()!=Math.multiplyExact(groups,Math.multiplyExact(capacity,dimension))
                || queries.getSize()!=Math.multiplyExact(groups,Math.multiplyExact(16,dimension)) || scores.getSize()!=Math.multiplyExact(groups,Math.multiplyExact(16,capacity)))
            throw new IllegalArgumentException("Invalid tensor-core sparse scores");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("gqa-scores")
                .withParameters(new Object[]{capacity,dimension,groups,control,keys,queries,alphaBeta,scores})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.WRITE_ONLY});
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
        void allocate() {
            if(handle!=0) return;
            try(var arena=Arena.ofConfined()) {
                var slot=arena.allocate(8,8);check((int)Native.CREATE.invokeExact(slot),"BF16/F32 cuBLASLt handle");handle=address(slot);
                check((int)Native.MALLOC.invokeExact(slot,WORKSPACE_BYTES),"BF16/F32 workspace");workspace=address(slot);CONTEXTS.incrementAndGet();
            } catch(Throwable error) {
                if(handle!=0) try {check((int)Native.DESTROY.invokeExact(handle),"BF16/F32 handle cleanup");handle=0;} catch(Throwable cleanup) {error.addSuppressed(cleanup);}
                throw failure(error);
            }
        }
    }
    private record Shape(int rows,int batch,int columns,int partition) {}
    private static final class Plan implements AutoCloseable {
        final Arena arena=Arena.ofShared();long desc,a,b,c;
        final MemorySegment algorithm=arena.allocate(96,8),attribute=arena.allocate(8,8);
        boolean counted;
        void integer(int attr,int value) throws Throwable {attribute.set(C_INT,0,value);check((int)Native.ATTR.invokeExact(desc,attr,attribute,4L),"BF16/F32 attribute "+attr);}
        void pointer(int attr,long value) throws Throwable {attribute.set(C_LONG,0,value);check((int)Native.ATTR.invokeExact(desc,attr,attribute,8L),"BF16/F32 pointer "+attr);}
        long layout(int type,long rows,long columns,long stride) throws Throwable {
            var out=arena.allocate(8,8);check((int)Native.LAYOUT.invokeExact(out,type,rows,columns,stride),"BF16/F32 matrix layout");return address(out);
        }
        Plan(Context context,Shape shape) {
            long preference=0;
            try {
                var slot=arena.allocate(8,8);check((int)Native.DESC.invokeExact(slot,68,0),"BF16/F32 matmul descriptor");desc=address(slot);
                integer(3,1);integer(4,0);integer(2,1); // TN, device alpha/beta.
                a=layout(14,shape.partition,shape.rows,shape.columns);
                b=layout(14,shape.partition,shape.batch,shape.columns);
                c=layout(0,shape.rows,shape.batch,shape.rows);
                check((int)Native.PREF.invokeExact(slot),"BF16/F32 preference");preference=address(slot);
                attribute.set(C_LONG,0,WORKSPACE_BYTES);check((int)Native.PREF_ATTR.invokeExact(preference,1,attribute,8L),"BF16/F32 workspace preference");
                for(int attr=5;attr<=8;attr++) {attribute.set(C_INT,0,16);check((int)Native.PREF_ATTR.invokeExact(preference,attr,attribute,4L),"BF16/F32 pointer alignment");}
                var count=arena.allocate(4,4);
                check((int)Native.HEURISTIC.invokeExact(context.handle,desc,a,b,c,c,preference,1,algorithm,count),"BF16/F32 heuristic");
                if(count.get(C_INT,0)!=1 || algorithm.get(C_INT,72)!=0 || algorithm.get(C_LONG,64)>WORKSPACE_BYTES)
                    throw new IllegalStateException("No cuBLASLt BF16/F32 algorithm for "+shape);
                counted=true;PLANS.incrementAndGet();
            } catch(Throwable error) {try {close();} catch(Exception cleanup) {error.addSuppressed(cleanup);}throw failure(error);}
            finally {if(preference!=0) try {check((int)Native.FREE_PREF.invokeExact(preference),"BF16/F32 preference close");} catch(Throwable error) {throw failure(error);}}
        }
        @Override public void close() {
            try {for(long layout:new long[]{a,b,c}) if(layout!=0) check((int)Native.FREE_LAYOUT.invokeExact(layout),"BF16/F32 layout close");
                a=b=c=0;if(desc!=0) {check((int)Native.FREE_DESC.invokeExact(desc),"BF16/F32 descriptor close");desc=0;}}
            catch(Throwable error) {throw failure(error);}
            finally {if(counted) {counted=false;PLANS.decrementAndGet();}arena.close();}
        }
    }
    @Override public String libraryName() {return NAME;}
    @Override public boolean canHandle(TornadoXPUDevice device) {return device instanceof TornadoNativeStreamSupport;}
    @Override public LibraryContext createContext(TornadoXPUDevice device,long id) {return new Context();}
    @Override public void prepare(LibraryTaskDescriptor descriptor,LibraryContext nativeContext) {
        if(descriptor.getFunctionName().equals("gqa-scores")) {((Context)nativeContext).allocate();return;}
        if(!descriptor.getFunctionName().equals("partitions")) throw new IllegalArgumentException("Unknown BF16/F32 function");
        var context=(Context)nativeContext;context.allocate();var args=descriptor.getParameters();
        var shape=new Shape((Integer)args[0],(Integer)args[1],(Integer)args[2],(Integer)args[7]);context.plans.computeIfAbsent(shape,key->new Plan(context,key));
    }
    @Override public void dispatch(String function,LibraryInvocation invocation) {
        if(function.equals("gqa-scores")) {dispatchGqaScores(invocation);return;}
        var context=(Context)invocation.getContext();var shape=new Shape((Integer)invocation.getArg(0),(Integer)invocation.getArg(1),(Integer)invocation.getArg(2),(Integer)invocation.getArg(7));
        var plan=context.plans.get(shape);if(plan==null) throw new IllegalStateException("BF16/F32 plan was not prepared");
        long alpha=invocation.getDevicePointer(5),stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try {
            for(int group=0;group<shape.columns/shape.partition;group++) {
                long output=invocation.getDevicePointer(6)+4L*group*shape.rows*shape.batch;
                check((int)Native.MATMUL.invokeExact(context.handle,plan.desc,alpha,invocation.getDevicePointer(3)+2L*shape.partition*group,plan.a,
                        invocation.getDevicePointer(4)+2L*shape.partition*group,plan.b,alpha+4,output,plan.c,output,plan.c,plan.algorithm,context.workspace,WORKSPACE_BYTES,stream),"BF16/F32 partition GEMM");
                CALLS.incrementAndGet();
            }
        } catch(Throwable error) {throw failure(error);}
    }
    private static void dispatchGqaScores(LibraryInvocation invocation) {
        var control=(IntArray)invocation.getArg(3);
        int capacity=(Integer)invocation.getArg(0),dimension=(Integer)invocation.getArg(1),groups=(Integer)invocation.getArg(2),count=control.get(2);
        int rows=(count+31)/32*32;if(count<=0 || rows>capacity) throw new IllegalArgumentException("Invalid sparse score count");
        var context=(Context)invocation.getContext();var shape=new Shape(rows,16,dimension,dimension);var plan=context.plans.computeIfAbsent(shape,key->new Plan(context,key));
        long alpha=invocation.getDevicePointer(6),stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try {
            for(int group=0;group<groups;group++) {
                long output=invocation.getDevicePointer(7)+4L*group*16*capacity;
                check((int)Native.MATMUL.invokeExact(context.handle,plan.desc,alpha,invocation.getDevicePointer(4)+2L*group*capacity*dimension,plan.a,
                        invocation.getDevicePointer(5)+2L*group*16*dimension,plan.b,alpha+4,output,plan.c,output,plan.c,plan.algorithm,context.workspace,WORKSPACE_BYTES,stream),"BF16/F32 sparse score GEMM");CALLS.incrementAndGet();
            }
        } catch(Throwable error) {throw failure(error);}
    }
    @Override public void destroyContext(LibraryContext nativeContext) {
        var context=(Context)nativeContext;
        try {for(var plan:context.plans.values()) plan.close();context.plans.clear();if(context.handle==0) return;
            check((int)Native.FREE.invokeExact(context.workspace),"BF16/F32 workspace close");context.workspace=0;
            check((int)Native.DESTROY.invokeExact(context.handle),"BF16/F32 handle close");context.handle=0;CONTEXTS.decrementAndGet();}
        catch(Throwable error) {throw failure(error);}
    }
}
