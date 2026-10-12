package org.beehive.jitllm.model;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;

/** A model that owns its device plans. Sessions still own and release their sequence state. */
public interface ManagedDeviceModel extends Model {
    StorageOptions defaultDeviceStorage();
    void validateDeviceOptions(ExecutionPolicy policy, StorageOptions storage);
    /** Describe the actual device path without advancing or resetting the sequence. */
    ExecutionInfo prepareDevice(State state);
    List<Integer> generateOnDevice(State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int budget, Sampler sampler, IntConsumer onToken);
}
