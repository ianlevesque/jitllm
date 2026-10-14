package org.beehive.jitllm.model;

import java.io.IOException;
import org.beehive.jitllm.model.vision.FlashNextVisionConfig;
import org.beehive.jitllm.model.vision.PreprocessedImage;

/**
 * A model that owns a vision tower the conversation encoder may splice into a prompt.
 *
 * <p>The encoder runs each image through {@link #preprocessImage(byte[])} once — which fixes
 * its grid, and so the merged token count its chat format expands to — and {@link
 * #imageEmbeddings(PreprocessedImage)} once for the rows that replace the token embedding at
 * the image positions. Implementations load tower weights lazily, so a text-only session
 * never pays for them.
 */
public interface VisionModel {

    /** The tower's dimensions and token ids, from the checkpoint's own configuration. */
    FlashNextVisionConfig visionConfig() throws IOException;

    /** Decode and patchify one encoded image; the result carries its patch grid. */
    PreprocessedImage preprocessImage(byte[] encodedBytes) throws IOException;

    /**
     * Tower embeddings for one preprocessed image: {@code [mergedTokenCount][hiddenSize]}
     * rows in merged-grid order, ready to stand in for the token embedding.
     */
    float[][] imageEmbeddings(PreprocessedImage image) throws IOException;
}
