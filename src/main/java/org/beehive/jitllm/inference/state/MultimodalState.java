package org.beehive.jitllm.inference.state;

/**
 * A session state that can accept a {@link MultimodalPromptPlan} for its next prompt.
 *
 * <p>The session layer stages the plan before generation begins; the engine then reads
 * tower rows and MRoPE positions from it as the prompt is ingested token by token, which
 * is why the generation loop's per-token signature needs no image parameters.
 */
public interface MultimodalState {

    /** Stages {@code plan} for the prompt about to be ingested from position zero. */
    void stageMultimodalPrompt(MultimodalPromptPlan plan);
}
