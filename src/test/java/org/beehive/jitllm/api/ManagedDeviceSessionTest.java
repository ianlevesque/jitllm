package org.beehive.jitllm.api;

import java.lang.reflect.*;
import java.nio.file.Path;
import java.util.*;
import org.beehive.jitllm.inference.state.*;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.*;
import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.beehive.jitllm.runtime.policy.*;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;
import static org.junit.Assert.*;

/** Ownership seam: prepare and generation must never build a legacy master plan. */
public final class ManagedDeviceSessionTest {
    private static final class Sequence implements Qwen4SequenceState {
        int position,closes,resets;
        public int position() {return position;}
        public void reset() {position=0;resets++;}
        public void close() {closes++;}
    }
    private static final class Fixture implements InvocationHandler {
        final Sequence sequence=new Sequence();int preparations,generations,validations;
        final Configuration config=(Configuration)Proxy.newProxyInstance(Configuration.class.getClassLoader(),new Class<?>[]{Configuration.class},(p,m,a)-> {
            if(m.isDefault()) return InvocationHandler.invokeDefault(p,m,a);
            return switch(m.getName()) {
                case "weightType","quantization"->"BF16";
                case "numberOfLayers","numberOfHeads","numberOfKeyValueHeads","numberOfHeadsKey"->1;
                case "contextLength","contextLengthModel"->16;
                case "rmsNormEps"->1e-6f;
                case "ropeTheta"->10000f;
                default->8;
            };
        });
        final ManagedDeviceModel model=(ManagedDeviceModel)Proxy.newProxyInstance(ManagedDeviceModel.class.getClassLoader(),new Class<?>[]{ManagedDeviceModel.class},this);
        final ExecutionInfo info=new ExecutionInfo("CUDA","fake device","fake runtime","single-token",1,"FP8","native","QSA",false,true);
        public Object invoke(Object proxy,Method method,Object[] args) throws Throwable {
            return switch(method.getName()) {
                case "createNewState"-> {var state=new Qwen4State(config,-1,sequence);state.latestToken=-1;yield state;}
                case "configuration"->config;
                case "getModelType"->ModelType.QWEN_4_EXP;
                case "weights"->(Weights)()->DataType.BF16;
                case "chatFormat","tokenizer"->null;
                case "defaultDeviceStorage"->StorageOptions.fp8();
                case "validateDeviceOptions"->{validations++;assertEquals(StorageOptions.fp8(),args[1]);yield null;}
                case "prepareDevice"->{preparations++;yield info;}
                case "generateOnDevice"->{generations++;sequence.position++;yield new ArrayList<>(List.of(7));}
                case "generateTokensGPU"->throw new AssertionError("Built a legacy GPU path");
                default->{if(method.isDefault()) yield InvocationHandler.invokeDefault(proxy,method,args);throw new UnsupportedOperationException(method.getName());}
            };
        }
    }
    @Test public void runtimeDelegatesWithoutCreatingAPlanAndReleasesItsSequence() {
        var f=new Fixture();
        try(var manager=KvCacheManager.sizedFor(1,16,16,32)) {
            var runtime=new LegacySessionRuntime(f.model,manager.acquire(16),ExecutionPolicy.builder().build(),StorageOptions.fp8());
            assertEquals(f.info,runtime.prepare());assertEquals(0,f.sequence.position());assertFalse(runtime.hasPlan());
            assertThrows(UnsupportedOperationException.class,runtime::plan);
            assertEquals(List.of(7),runtime.generateOnGpu(f.model,0,List.of(1),Set.of(),2,null,null));
            assertEquals(1,f.generations);assertEquals(1,f.sequence.position());
            runtime.reset();assertEquals(0,f.sequence.position());assertEquals(-1,runtime.executionState().latestToken);
            runtime.close();runtime.close();assertEquals(1,f.sequence.closes);
            try(var recycled=manager.acquire(16)) {assertEquals(16,recycled.tokenCapacity());}
        }
    }
    @Test public void facadePreparesManagedDeviceAndValidatesBeforeStateAllocation() {
        var f=new Fixture();
        try(var model=new DelegatingModel(f.model,Path.of("fake"),true,ExecutionPolicy.builder().build(),StorageOptions.fp8())) {
            assertEquals(1,f.validations);
            try(var session=model.newSession()) {
                assertEquals(f.info,session.prepare());assertEquals(f.info,session.prepare());assertEquals(0,session.position());
                session.reset();assertEquals(1,f.sequence.resets);
                assertThrows(IllegalStateException.class,model::close);
            }
            assertEquals(1,f.sequence.closes);
        }
    }
    @Test public void explicitStorageIsDistinguishedFromTheModelDefault() {
        assertTrue(ModelOptions.builder().storageOptions(StorageOptions.fp16()).build().hasExplicitStorageOptions());
        assertTrue(ModelOptions.builder().storageOptions(StorageOptions.fp8()).build().hasExplicitStorageOptions());
        assertEquals(DataType.F8_E4M3,StorageOptions.fp8().keyValueRepresentation());
        assertFalse(StorageOptions.fp8().usesFp16KeyValueCache());
    }
}
