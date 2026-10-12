package org.beehive.jitllm.api;

import java.lang.reflect.*;
import java.io.IOException;
import org.beehive.jitllm.inference.state.Qwen4SequenceState;
import org.beehive.jitllm.inference.state.Qwen4State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.beehive.jitllm.runtime.policy.*;
import org.junit.Test;
import static org.junit.Assert.*;

/** Session state can own device resources even when there is no legacy master plan. */
public final class LegacySessionResourceTest {
    private static final class Resource implements Qwen4SequenceState {
        int closes;
        boolean fail;
        public int position() {return 0;}
        public void reset() {}
        public void close() throws IOException {closes++;if(fail) throw new IOException("close failure");}
    }
    private static LegacySessionRuntime runtime(KvCacheManager manager,Resource resource) {
        Configuration config=(Configuration)Proxy.newProxyInstance(Configuration.class.getClassLoader(),new Class<?>[]{Configuration.class},(proxy,method,args)-> {
            if(method.isDefault()) return InvocationHandler.invokeDefault(proxy,method,args);
            return switch(method.getName()) {
                case "weightType"->"BF16";
                case "numberOfLayers","numberOfHeads","numberOfKeyValueHeads","numberOfHeadsKey"->1;
                case "contextLength","contextLengthModel"->16;
                case "rmsNormEps"->1e-6f;
                case "ropeTheta"->10000f;
                default->8;
            };
        });
        Model model=(Model)Proxy.newProxyInstance(Model.class.getClassLoader(),new Class<?>[]{Model.class},(proxy,method,args)-> {
            if(method.getName().equals("createNewState")) return new Qwen4State(config,-1,resource);
            if(method.getName().equals("configuration")) return config;
            if(method.isDefault()) return InvocationHandler.invokeDefault(proxy,method,args);
            throw new UnsupportedOperationException(method.getName());
        });
        return new LegacySessionRuntime(model,manager.acquire(16),ExecutionPolicy.fromSystemProperties(),StorageOptions.fp16());
    }
    @Test public void closeReleasesOwnedStateOnceAndReturnsTheLease() {
        try(var manager=KvCacheManager.sizedFor(1,16,16,32)) {
            var resource=new Resource();var runtime=runtime(manager,resource);
            runtime.close();runtime.close();assertEquals(1,resource.closes);
            try(var recycled=manager.acquire(16)) {assertEquals(16,recycled.tokenCapacity());}
        }
    }
    @Test public void failingStateCloseStillReturnsTheLease() {
        try(var manager=KvCacheManager.sizedFor(1,16,16,32)) {
            var resource=new Resource();resource.fail=true;var runtime=runtime(manager,resource);
            assertThrows(IllegalStateException.class,runtime::close);assertEquals(1,resource.closes);
            try(var recycled=manager.acquire(16)) {assertEquals(16,recycled.tokenCapacity());}
        }
    }
}
