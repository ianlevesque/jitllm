package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
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

/** Staged SM120 XQA decode through the exported TVM-FFI C ABI, without a Python runtime. */
public final class FlashNextXqaLibrary implements TornadoLibraryProvider {
    public static final String NAME="jitllm/xqa";
    public static final long WORKSPACE_BYTES=16L<<20,SEMAPHORE_BYTES=8L<<20;
    private static final AtomicLong CONTEXTS=new AtomicLong(),CALLS=new AtomicLong();
    public record Statistics(long contexts,long workspaceBytes,long calls) {}
    public static Statistics statistics() {return new Statistics(CONTEXTS.get(),CONTEXTS.get()*WORKSPACE_BYTES,CALLS.get());}
    public static LibraryTaskDescriptor decode(int capacity,int dimension,int heads,int kvHeads,IntArray control,
            BFloat16Array query,BFloat16Array keys,BFloat16Array values,BFloat16Array output,IntArray pages) {
        if(capacity<=0 || capacity>2112 || capacity%64!=0 || dimension!=256 || heads!=24 || kvHeads!=2 || control.getSize()<6
                || query.getSize()!=heads*dimension || output.getSize()!=query.getSize()
                || keys.getSize()!=capacity*kvHeads*dimension || values.getSize()!=keys.getSize() || pages.getSize()!=capacity/64)
            throw new IllegalArgumentException("XQA requires the Flash-Next BF16 page-64 geometry");
        return new LibraryTaskDescriptor().withLibrary(NAME).withFunction("decode")
                .withParameters(new Object[]{capacity,dimension,heads,kvHeads,control,query,keys,values,output,pages})
                .withAccess(new Access[]{Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,
                        Access.READ_ONLY,Access.READ_ONLY,Access.READ_ONLY,Access.WRITE_ONLY,Access.READ_ONLY});
    }
    private static final class Native {
        static final SymbolLookup CUDA=required("libcudart.so.13","libcudart.so"),DRIVER=required("libcuda.so.1");
        static SymbolLookup required(String... names) {var library=FFMSupport.loadLibrary(names);if(library==null) throw new IllegalStateException("Missing XQA CUDA dependency");return library;}
        static MethodHandle call(SymbolLookup lookup,String name,MemoryLayout result,MemoryLayout... args) {
            var handle=FFMSupport.downcall(lookup,FunctionDescriptor.of(result,args),name);
            if(handle==null) throw new IllegalStateException("Missing XQA native symbol "+name);return handle;
        }
        static final MethodHandle MALLOC=call(CUDA,"cudaMalloc",C_INT,C_POINTER,C_LONG),FREE=call(CUDA,"cudaFree",C_INT,C_LONG),
                MEMSET=call(CUDA,"cudaMemset",C_INT,C_LONG,C_INT,C_LONG),DEVICE=call(DRIVER,"cuCtxGetDevice",C_INT,C_POINTER),
                ATTRIBUTE=call(DRIVER,"cuDeviceGetAttribute",C_INT,C_POINTER,C_INT,C_INT),
                DLOPEN=call(Linker.nativeLinker().defaultLookup(),"dlopen",C_POINTER,C_POINTER,C_INT),
                DLCLOSE=call(Linker.nativeLinker().defaultLookup(),"dlclose",C_INT,C_POINTER);
    }
    private static void check(int status,String action) {if(status!=0) throw new IllegalStateException(action+" failed with native status "+status);}
    private static RuntimeException failure(Throwable error) {if(error instanceof RuntimeException r) return r;if(error instanceof Error e) throw e;return new IllegalStateException(error);}
    private static Path library(String variable) {
        String value=System.getenv(variable);
        if(value==null || value.isBlank()) throw new IllegalStateException("Prepare native QSA dependencies and export "+variable);
        var path=Path.of(value);if(!path.isAbsolute() || !Files.isRegularFile(path)) throw new IllegalStateException("Missing XQA library "+path);return path;
    }
    private static final class Context implements LibraryContext {
        final Arena libraries=Arena.ofShared();
        long workspace;
        MemorySegment globalFfi=MemorySegment.NULL;
        MethodHandle run,setStream,errorMove,decref;
        int device,smCount;
        boolean counted,closed;
        void prepare() {
            if(counted) return;if(closed) throw new IllegalStateException("XQA context is closed");
            try(var temporary=Arena.ofConfined()) {
                var slot=temporary.allocate(8,8);check((int)Native.DEVICE.invokeExact(slot),"XQA CUDA device");device=slot.get(C_INT,0);
                check((int)Native.ATTRIBUTE.invokeExact(slot,75,device),"XQA compute major");int major=slot.get(C_INT,0);
                check((int)Native.ATTRIBUTE.invokeExact(slot,76,device),"XQA compute minor");int minor=slot.get(C_INT,0);
                if(major!=12 || minor!=0) throw new IllegalStateException("Native XQA binding is qualified only for SM120");
                check((int)Native.ATTRIBUTE.invokeExact(slot,16,device),"XQA SM count");smCount=slot.get(C_INT,0);
                var ffi=library("JITLLM_TVM_FFI_LIBRARY");var xqa=library("JITLLM_FLASHINFER_XQA_LIBRARY");
                String ffiHash=System.getenv("JITLLM_TVM_FFI_SHA256");
                if(ffiHash==null || !ffiHash.equals(HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(ffi)))))
                    throw new IllegalStateException("Use the prepared JVM-compatible TVM-FFI library and its hash");
                String expected=System.getenv("JITLLM_FLASHINFER_XQA_SHA256");
                if(expected!=null && !expected.equals(HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(xqa)))))
                    throw new IllegalStateException("Prepared XQA library hash changed");
                // The JIT module references TVM-FFI symbols without a DT_NEEDED entry.
                globalFfi=(MemorySegment)Native.DLOPEN.invokeExact(temporary.allocateFrom(ffi.toString()),258); // RTLD_NOW | RTLD_GLOBAL
                if(globalFfi.address()==0) throw new IllegalStateException("Cannot load TVM-FFI globally");
                var ffiLookup=SymbolLookup.libraryLookup(ffi,libraries);var xqaLookup=SymbolLookup.libraryLookup(xqa,libraries);
                run=Native.call(xqaLookup,"__tvm_ffi_xqa_wrapper",C_INT,C_POINTER,C_POINTER,C_INT,C_POINTER);
                setStream=Native.call(ffiLookup,"TVMFFIEnvSetStream",C_INT,C_INT,C_INT,C_POINTER,C_POINTER);
                errorMove=Native.call(ffiLookup,"TVMFFIErrorMoveFromRaised",C_INT,C_POINTER);
                decref=Native.call(ffiLookup,"TVMFFIObjectDecRef",C_INT,C_POINTER);
                check((int)Native.MALLOC.invokeExact(slot,WORKSPACE_BYTES),"XQA workspace allocation");workspace=slot.get(C_LONG,0);
                check((int)Native.MEMSET.invokeExact(workspace,0,WORKSPACE_BYTES),"XQA semaphore initialization");
                counted=true;CONTEXTS.incrementAndGet();
            } catch(Throwable error) {try {close();} catch(Throwable cleanup) {error.addSuppressed(cleanup);}throw failure(error);}
        }
        String error(Arena arena) throws Throwable {
            var slot=arena.allocate(8,8);check((int)errorMove.invokeExact(slot),"Read XQA error");var object=slot.get(C_POINTER,0);
            if(object.address()==0) return "XQA failed without an error object";
            try {
                var cell=object.reinterpret(72);long size=cell.get(C_LONG,48);
                if(size<0 || size>1048576) return "XQA error message has invalid size";
                return new String(cell.get(C_POINTER,40).reinterpret(size).toArray(ValueLayout.JAVA_BYTE),StandardCharsets.UTF_8);
            } finally {check((int)decref.invokeExact(object),"Release XQA error");}
        }
        void close() {
            if(closed) return;closed=true;Throwable failure=null;
            try {if(workspace!=0) {check((int)Native.FREE.invokeExact(workspace),"XQA workspace release");workspace=0;}}
            catch(Throwable error) {failure=error;}
            try {libraries.close();} catch(Throwable error) {if(failure==null) failure=error;else failure.addSuppressed(error);}
            try {if(globalFfi.address()!=0) {check((int)Native.DLCLOSE.invokeExact(globalFfi),"TVM-FFI global release");globalFfi=MemorySegment.NULL;}}
            catch(Throwable error) {if(failure==null) failure=error;else failure.addSuppressed(error);}
            if(counted) {counted=false;CONTEXTS.decrementAndGet();}if(failure!=null) throw failure(failure);
        }
    }
    // TVMFFIAny is {int32 type, uint32 padding, union64 value}; DLTensor is 48 bytes on this ABI.
    private static void integer(MemorySegment args,int index,int type,long value) {args.set(C_INT,index*16L,type);args.set(C_LONG,index*16L+8,value);}
    private static void real(MemorySegment args,int index,double value) {args.set(C_INT,index*16L,3);args.set(C_DOUBLE,index*16L+8,value);}
    private static void tensor(Arena arena,MemorySegment args,int index,long data,int device,int dtype,long... shape) {
        var dimensions=arena.allocate(shape.length*8L,8);var strides=arena.allocate(shape.length*8L,8);long stride=1;
        for(int i=shape.length-1;i>=0;i--) {dimensions.set(C_LONG,i*8L,shape[i]);strides.set(C_LONG,i*8L,stride);stride=Math.multiplyExact(stride,shape[i]);}
        var tensor=arena.allocate(48,8);tensor.set(C_LONG,0,data);tensor.set(C_INT,8,2);tensor.set(C_INT,12,device);
        tensor.set(C_INT,16,shape.length);tensor.set(C_INT,20,dtype);tensor.set(C_POINTER,24,dimensions);tensor.set(C_POINTER,32,strides);
        integer(args,index,7,tensor.address());
    }
    @Override public String libraryName() {return NAME;}
    @Override public boolean canHandle(TornadoXPUDevice device) {return device instanceof TornadoNativeStreamSupport;}
    @Override public LibraryContext createContext(TornadoXPUDevice device,long id) {return new Context();}
    @Override public void prepare(LibraryTaskDescriptor descriptor,LibraryContext context) {
        if(!descriptor.getFunctionName().equals("decode")) throw new IllegalArgumentException("Unknown XQA function");((Context)context).prepare();
    }
    @Override public void dispatch(String function,LibraryInvocation invocation) {
        if(invocation.isCapturing()) throw new UnsupportedOperationException("XQA graph capture remains unqualified");
        var control=(IntArray)invocation.getArg(4);if(control.get(3)!=0) return;
        int capacity=(Integer)invocation.getArg(0),dimension=(Integer)invocation.getArg(1),heads=(Integer)invocation.getArg(2),kvHeads=(Integer)invocation.getArg(3);
        if(control.get(2)<=0 || control.get(2)>capacity) throw new IllegalArgumentException("Invalid XQA selected count");
        var context=(Context)invocation.getContext();long stream=((TornadoNativeStreamSupport)invocation.getDevice()).getNativeStream(invocation.getExecutionPlanId());
        try(var arena=Arena.ofConfined()) {
            var args=arena.allocate(26*16L,8);int d=context.device;
            integer(args,0,2,0);integer(args,1,1,context.smCount);integer(args,2,1,kvHeads);integer(args,3,1,0);real(args,4,1);
            tensor(arena,args,6,invocation.getDevicePointer(8),d,0x11004,1,1,heads,dimension);real(args,7,1);
            tensor(arena,args,8,invocation.getDevicePointer(5),d,0x11004,1,1,heads,dimension);
            tensor(arena,args,10,invocation.getDevicePointer(6),d,0x11004,capacity/64,64,kvHeads,dimension);
            tensor(arena,args,11,invocation.getDevicePointer(7),d,0x11004,capacity/64,64,kvHeads,dimension);
            tensor(arena,args,14,invocation.getDevicePointer(9),d,0x12000,1,capacity/64);integer(args,15,1,capacity);
            tensor(arena,args,16,invocation.getDevicePointer(4)+8,d,0x12000,1,1);integer(args,17,1,1);real(args,18,1);integer(args,20,1,1);
            tensor(arena,args,23,context.workspace,d,0x10801,SEMAPHORE_BYTES);
            tensor(arena,args,24,context.workspace+SEMAPHORE_BYTES,d,0x10801,WORKSPACE_BYTES-SEMAPHORE_BYTES);integer(args,25,2,0);
            var previous=arena.allocate(8,8);check((int)context.setStream.invokeExact(2,d,MemorySegment.ofAddress(stream),previous),"Set XQA stream");
            try {
                var result=arena.allocate(16,8);int status=(int)context.run.invokeExact(MemorySegment.NULL,args,26,result);
                if(status!=0) throw new IllegalStateException(context.error(arena));CALLS.incrementAndGet();
            } finally {check((int)context.setStream.invokeExact(2,d,previous.get(C_POINTER,0),MemorySegment.NULL),"Restore XQA stream");}
        } catch(Throwable error) {throw failure(error);}
    }
    @Override public void destroyContext(LibraryContext context) {((Context)context).close();}
}
