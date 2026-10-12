package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ConcurrentHashMap;
import uk.ac.manchester.tornado.api.common.*;
import uk.ac.manchester.tornado.api.types.arrays.*;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.ffm.FFMSupport;
import uk.ac.manchester.tornado.runtime.library.spi.*;
import static uk.ac.manchester.tornado.runtime.ffm.FFMSupport.*;

/** Native E4M3 tensor-core dot products in K=128 partitions. CUDA applies the original
 * 128x128 weight and 128-wide activation scales after each partition, without BF16 weights.
 * CUDA 13's direct FP32 block-scale descriptors are restricted to Hopper (SM90).
 */
public final class FlashNextFp8Library implements TornadoLibraryProvider {
    public static final String NAME="jitllm/fp8-block";
    public static final long WORKSPACE_BYTES=1L<<20;
    private static final AtomicLong CONTEXTS=new AtomicLong(),PLANS=new AtomicLong(),CALLS=new AtomicLong();
    public record Statistics(long contexts,long plans,long workspaceBytes,long calls) {}
    public static Statistics statistics() {return new Statistics(CONTEXTS.get(),PLANS.get(),CONTEXTS.get()*WORKSPACE_BYTES,CALLS.get());}
    public static LibraryTaskDescriptor partitions(int rows,int batch,int columns,ByteArray weights,
            ByteArray input,FloatArray alphaBeta,FloatArray partials) {
        if(rows<=0 || batch<=0 || columns<=0 || rows%16!=0 || batch%16!=0 || columns%128!=0
                || alphaBeta.getSize()<3 || weights.getSize()!=Math.multiplyExact(rows,columns)
                || input.getSize()!=Math.multiplyExact(batch,columns)
                || partials.getSize()!=Math.multiplyExact(Math.multiplyExact(rows,batch),columns/128))
            throw new IllegalArgumentException("Invalid block FP8 GEMM geometry");
        Access[] access=new Access[7];Arrays.fill(access,Access.READ_ONLY);access[6]=Access.WRITE_ONLY;
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("partitions")
                .withParameters(new Object[]{rows,batch,columns,weights,input,alphaBeta,partials}).withAccess(access);
    }
    private record Registered(TornadoXPUDevice device,long owner,int rows,int columns,long weights) {}
    private static final Map<Integer,Registered> REGISTERED=new ConcurrentHashMap<>();
    public static int registeredWeights() {return REGISTERED.size();}
    public static LibraryTaskDescriptor register(int id,int rows,int columns,ByteArray weights) {
        if(id<=0 || rows<=0 || columns<=0 || rows%128!=0 || columns%128!=0
                || weights.getSize()!=Math.multiplyExact(rows,columns))
            throw new IllegalArgumentException("Invalid FP8 expert registration");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("register")
                .withParameters(new Object[]{id,rows,columns,weights})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY});
    }
    public static LibraryTaskDescriptor experts(int rows,int batch,int columns,int count,IntArray ids,
            ByteArray input,FloatArray control,FloatArray partials,int inputStride) {
        if(rows<=0 || batch<=0 || columns<=0 || rows%128!=0 || batch%16!=0 || columns%128!=0
                || count<=0 || ids.getSize()!=count || control.getSize()<3
                || (inputStride!=0 && inputStride!=Math.multiplyExact(batch,columns))
                || input.getSize()!=Math.multiplyExact(inputStride==0 ? 1 : count,Math.multiplyExact(batch,columns))
                || partials.getSize()!=Math.multiplyExact(count,Math.multiplyExact(Math.multiplyExact(rows,batch),columns/128)))
            throw new IllegalArgumentException("Invalid selected FP8 expert geometry");
        Access[] access=new Access[9];Arrays.fill(access,Access.READ_ONLY);access[7]=Access.WRITE_ONLY;
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("experts")
                .withParameters(new Object[]{rows,batch,columns,count,ids,input,control,partials,inputStride}).withAccess(access);
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
        final Set<Integer> registrations=new HashSet<>();
        void allocate() {
            if(handle!=0) return;
            try(var arena=Arena.ofConfined()) {
                var slot=arena.allocate(8,8);check((int)Native.CREATE.invokeExact(slot),"FP8 cuBLASLt handle");handle=address(slot);
                check((int)Native.MALLOC.invokeExact(slot,WORKSPACE_BYTES),"FP8 workspace");workspace=address(slot);CONTEXTS.incrementAndGet();
            } catch(Throwable error) {
                if(handle!=0) try {check((int)Native.DESTROY.invokeExact(handle),"FP8 handle cleanup");handle=0;} catch(Throwable cleanup) {error.addSuppressed(cleanup);}
                throw failure(error);
            }
        }
    }
    private record Shape(int rows,int batch,int columns) {}
    private static final class Plan implements AutoCloseable {
        final Arena arena=Arena.ofShared();long desc,a,b,c;
        final MemorySegment algorithm=arena.allocate(96,8),attribute=arena.allocate(8,8);
        boolean counted;
        void integer(int attr,int value) throws Throwable {attribute.set(C_INT,0,value);check((int)Native.ATTR.invokeExact(desc,attr,attribute,4L),"FP8 attribute "+attr);}
        void pointer(int attr,long value) throws Throwable {attribute.set(C_LONG,0,value);check((int)Native.ATTR.invokeExact(desc,attr,attribute,8L),"FP8 pointer "+attr);}
        long layout(int type,long rows,long columns,long stride) throws Throwable {
            var out=arena.allocate(8,8);check((int)Native.LAYOUT.invokeExact(out,type,rows,columns,stride),"FP8 matrix layout");return address(out);
        }
        Plan(Context context,Shape shape) {
            long preference=0;
            try {
                var slot=arena.allocate(8,8);check((int)Native.DESC.invokeExact(slot,68,0),"FP8 matmul descriptor");desc=address(slot);
                integer(3,1);integer(4,0);integer(2,1); // TN, device alpha/beta.
                attribute.set(C_CHAR,0,(byte)0);check((int)Native.ATTR.invokeExact(desc,25,attribute,1L),"FP8 full accumulation");
                pointer(17,context.workspace);pointer(18,context.workspace);
                a=layout(28,128,shape.rows,shape.columns);
                b=layout(28,128,shape.batch,shape.columns);
                c=layout(0,shape.rows,shape.batch,shape.rows);
                check((int)Native.PREF.invokeExact(slot),"FP8 preference");preference=address(slot);
                attribute.set(C_LONG,0,WORKSPACE_BYTES);check((int)Native.PREF_ATTR.invokeExact(preference,1,attribute,8L),"FP8 workspace preference");
                for(int attr=5;attr<=8;attr++) {attribute.set(C_INT,0,16);check((int)Native.PREF_ATTR.invokeExact(preference,attr,attribute,4L),"FP8 pointer alignment");}
                var count=arena.allocate(4,4);
                check((int)Native.HEURISTIC.invokeExact(context.handle,desc,a,b,c,c,preference,1,algorithm,count),"FP8 heuristic");
                if(count.get(C_INT,0)!=1 || algorithm.get(C_INT,72)!=0 || algorithm.get(C_LONG,64)>WORKSPACE_BYTES)
                    throw new IllegalStateException("No cuBLASLt FP8 algorithm for "+shape);
                counted=true;PLANS.incrementAndGet();
            } catch(Throwable error) {try {close();} catch(Exception cleanup) {error.addSuppressed(cleanup);}throw failure(error);}
            finally {if(preference!=0) try {check((int)Native.FREE_PREF.invokeExact(preference),"FP8 preference close");} catch(Throwable error) {throw failure(error);}}
        }
        @Override public void close() {
            try {for(long layout:new long[]{a,b,c}) if(layout!=0) check((int)Native.FREE_LAYOUT.invokeExact(layout),"FP8 layout close");
                a=b=c=0;if(desc!=0) {check((int)Native.FREE_DESC.invokeExact(desc),"FP8 descriptor close");desc=0;}}
            catch(Throwable error) {throw failure(error);}
            finally {if(counted) {counted=false;PLANS.decrementAndGet();}arena.close();}
        }
    }
    @Override public String libraryName() {return NAME;}
    @Override public boolean canHandle(TornadoXPUDevice device) {return device instanceof TornadoNativeStreamSupport;}
    @Override public LibraryContext createContext(TornadoXPUDevice device,long id) {return new Context();}
    @Override public void prepare(LibraryTaskDescriptor descriptor,LibraryContext nativeContext) {
        if(descriptor.getFunctionName().equals("register")) return;
        if(!descriptor.getFunctionName().equals("partitions") && !descriptor.getFunctionName().equals("experts")) throw new IllegalArgumentException("Unknown FP8 function");
        var context=(Context)nativeContext;context.allocate();var args=descriptor.getParameters();
        var shape=new Shape((Integer)args[0],(Integer)args[1],(Integer)args[2]);context.plans.computeIfAbsent(shape,key->new Plan(context,key));
    }
    @Override public void dispatch(String function,LibraryInvocation invocation) {
        if(function.equals("register")) {
            int id=(Integer)invocation.getArg(0);var context=(Context)invocation.getContext();
            var record=new Registered(invocation.getDevice(),invocation.getExecutionPlanId(),(Integer)invocation.getArg(1),(Integer)invocation.getArg(2),invocation.getDevicePointer(3));
            var previous=REGISTERED.putIfAbsent(id,record);if(previous!=null && !previous.equals(record)) throw new IllegalStateException("Conflicting FP8 resource ID");
            context.registrations.add(id);return;
        }
        if(function.equals("experts")) {dispatchExperts(invocation);return;}
        var context=(Context)invocation.getContext();var shape=new Shape((Integer)invocation.getArg(0),(Integer)invocation.getArg(1),(Integer)invocation.getArg(2));
        var plan=context.plans.get(shape);if(plan==null) throw new IllegalStateException("FP8 plan was not prepared");
        long alpha=invocation.getDevicePointer(5),stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try {
            plan.pointer(17,alpha+8);plan.pointer(18,alpha+8);
            for(int group=0;group<shape.columns/128;group++) {
                long output=invocation.getDevicePointer(6)+4L*group*shape.rows*shape.batch;
                check((int)Native.MATMUL.invokeExact(context.handle,plan.desc,alpha,invocation.getDevicePointer(3)+128L*group,plan.a,
                        invocation.getDevicePointer(4)+128L*group,plan.b,alpha+4,output,plan.c,output,plan.c,plan.algorithm,context.workspace,WORKSPACE_BYTES,stream),"FP8 partition GEMM");
                CALLS.incrementAndGet();
            }
        } catch(Throwable error) {throw failure(error);}
    }
    private static void dispatchExperts(LibraryInvocation invocation) {
        var context=(Context)invocation.getContext();var shape=new Shape((Integer)invocation.getArg(0),(Integer)invocation.getArg(1),(Integer)invocation.getArg(2));
        var plan=context.plans.get(shape);int count=(Integer)invocation.getArg(3),stride=(Integer)invocation.getArg(8);
        var ids=(IntArray)invocation.getArg(4);long alpha=invocation.getDevicePointer(6);
        long stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try {
            plan.pointer(17,alpha+8);plan.pointer(18,alpha+8);
            for(int expert=0;expert<count;expert++) {
                var weight=REGISTERED.get(ids.get(expert));
                if(weight==null || !weight.device.equals(invocation.getDevice()) || weight.rows!=shape.rows || weight.columns!=shape.columns)
                    throw new IllegalStateException("Invalid FP8 expert resource "+ids.get(expert));
                for(int group=0;group<shape.columns/128;group++) {
                    long output=invocation.getDevicePointer(7)+4L*(expert*(shape.columns/128)+group)*shape.rows*shape.batch;
                    check((int)Native.MATMUL.invokeExact(context.handle,plan.desc,alpha,weight.weights+128L*group,plan.a,
                            invocation.getDevicePointer(5)+(long)expert*stride+128L*group,plan.b,alpha+4,output,plan.c,output,plan.c,
                            plan.algorithm,context.workspace,WORKSPACE_BYTES,stream),"FP8 selected expert partition");CALLS.incrementAndGet();
                }
            }
        } catch(Throwable error) {throw failure(error);}
    }
    @Override public void destroyContext(LibraryContext nativeContext) {
        var context=(Context)nativeContext;
        try {for(int id:context.registrations) REGISTERED.remove(id);context.registrations.clear();for(var plan:context.plans.values()) plan.close();context.plans.clear();if(context.handle==0) return;
            check((int)Native.FREE.invokeExact(context.workspace),"FP8 workspace close");context.workspace=0;
            check((int)Native.DESTROY.invokeExact(context.handle),"FP8 handle close");context.handle=0;CONTEXTS.decrementAndGet();}
        catch(Throwable error) {throw failure(error);}
    }
}
