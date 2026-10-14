package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

/** The sampling controls beyond temperature/top-p: disabled by default, validated at build. */
public class GenerationRequestSamplingTest {

    @Test
    public void defaultsDisableEveryAddedControl() {
        GenerationRequest request = GenerationRequest.of("Why is the sky blue?");
        assertEquals(0, request.topK());
        assertEquals(0.0f, request.minP(), 0.0f);
        assertEquals(0.0f, request.frequencyPenalty(), 0.0f);
        assertEquals(0.0f, request.presencePenalty(), 0.0f);
        assertEquals(1.0f, request.repetitionPenalty(), 0.0f);
    }

    @Test
    public void builderValuesRoundTrip() {
        GenerationRequest request =
                GenerationRequest.builder()
                        .prompt("x")
                        .topK(20)
                        .minP(0.05f)
                        .frequencyPenalty(0.3f)
                        .presencePenalty(-0.1f)
                        .repetitionPenalty(1.1f)
                        .build();
        assertEquals(20, request.topK());
        assertEquals(0.05f, request.minP(), 0.0f);
        assertEquals(0.3f, request.frequencyPenalty(), 0.0f);
        assertEquals(-0.1f, request.presencePenalty(), 0.0f);
        assertEquals(1.1f, request.repetitionPenalty(), 0.0f);
    }

    @Test
    public void minusOneTopKIsAcceptedAsDisabled() {
        GenerationRequest request = GenerationRequest.builder().prompt("x").topK(-1).build();
        assertEquals(-1, request.topK());
    }

    @Test
    public void invalidSamplingValuesAreRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").topK(-2).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").minP(-0.1f).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").minP(1.1f).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").minP(Float.NaN).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").frequencyPenalty(Float.NaN).build());
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        GenerationRequest.builder()
                                .prompt("x")
                                .presencePenalty(Float.POSITIVE_INFINITY)
                                .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").repetitionPenalty(0f).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").repetitionPenalty(-1f).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").repetitionPenalty(Float.NaN).build());
    }
}
