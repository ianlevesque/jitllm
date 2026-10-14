package org.beehive.jitllm.api;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import javax.imageio.ImageIO;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.MultimodalPromptPlan;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.VisionModel;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.format.Qwen4ChatFormat;
import org.beehive.jitllm.model.vision.FlashNextImagePreprocessor;
import org.beehive.jitllm.model.vision.FlashNextVisionConfig;
import org.beehive.jitllm.model.vision.ImageGrid;
import org.beehive.jitllm.model.vision.PreprocessedImage;
import org.beehive.jitllm.tokenizer.Qwen4Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * The encoder/session splice for image turns: real preprocessing (so grids are the
 * processor's own), a fake tower whose rows are numbered, and the synthetic byte tokenizer
 * of {@link Qwen4ChatFormatTemplateTest}. Full-conversation expectations are the checkpoint
 * template rendered by Jinja2, with each single {@code <|image_pad|>} placeholder expanded
 * to its grid's merged token count as the image processor expands it.
 */
public class VisionConversationEncoderTest {

    private Path directory;
    private Qwen4Tokenizer tokenizer;
    private Qwen4ChatFormat format;
    private FakeVisionModel model;
    private ConversationEncoder encoder;

    @Before
    public void setUp() throws IOException {
        directory = Files.createTempDirectory("qwen4-vision-encoder-test-");
        Files.writeString(
                directory.resolve("tokenizer.json"), Qwen4ChatFormatTemplateTest.syntheticTokenizerJson());
        tokenizer = Qwen4Tokenizer.load(directory);
        format = new Qwen4ChatFormat(tokenizer);
        var special = tokenizer.getSpecialTokens();
        FlashNextVisionConfig config =
                new FlashNextVisionConfig(
                        1, 16, 2, 32, 3, 16, 2, 2, 4, 2304, "gelu_pytorch_tanh",
                        new float[] {0.5f, 0.5f, 0.5f}, new float[] {0.5f, 0.5f, 0.5f},
                        1024, 16_777_216,
                        special.get("<|image_pad|>"), -1,
                        special.get("<|vision_start|>"), special.get("<|vision_end|>"));
        model = new FakeVisionModel(format, config);
        encoder = new ConversationEncoder(model, ThinkingMode.DISABLED, ReasoningEffort.DEFAULT);
    }

    @After
    public void tearDown() throws IOException {
        Files.deleteIfExists(directory.resolve("tokenizer.json"));
        Files.deleteIfExists(directory);
    }

    // ---- full-conversation rendering -----------------------------------------------------------

    @Test
    public void anImageConversationMatchesTheJinjaRenderingExpanded() throws IOException {
        var encoded =
                encoder.encodeConversation(
                        List.of(
                                new ChatMessage(
                                        ChatRole.USER,
                                        List.of(
                                                new ChatContent.Image(png(96, 64)),
                                                new ChatContent.Text("What is in this picture?")))),
                        List.of());
        assertTrue(encoded.hasImages());
        assertEquals(
                EXPECTED_IMAGE_THEN_TEXT_DISABLED.replace(
                        "<|image_pad|>", "<|image_pad|>".repeat(6)),
                tokenizer.decode(encoded.tokens()));
        assertEquals(1, model.preprocessCalls);
        assertEquals(1, model.towerCalls);
    }

    // ---- splice ordering ----------------------------------------------------------------------

    @Test
    public void twoImagesAreSplicedInContentOrderWithTheirOwnRows() throws IOException {
        var encoded =
                encoder.encodeConversation(
                        List.of(
                                new ChatMessage(
                                        ChatRole.USER,
                                        List.of(
                                                new ChatContent.Text("Look: "),
                                                new ChatContent.Image(png(96, 64)),
                                                new ChatContent.Text(" and "),
                                                new ChatContent.Image(png(64, 64)),
                                                new ChatContent.Text("?")))),
                        List.of());
        List<Integer> tokens = encoded.tokens();
        MultimodalPromptPlan plan = encoded.imagePlan();
        assertEquals(2, plan.spans().size());
        assertEquals(2, model.preprocessCalls);
        assertEquals(2, model.towerCalls);
        assertEquals(new ImageGrid(1, 4, 6), plan.spans().get(0).grid());
        assertEquals(new ImageGrid(1, 4, 4), plan.spans().get(1).grid());

        int imagePad = tokenizer.getSpecialTokens().get("<|image_pad|>");
        int[] starts = {plan.spans().get(0).start(), plan.spans().get(1).start()};
        int[] counts = {6, 4};
        for (int s = 0; s < 2; s++) {
            for (int i = 0; i < counts[s]; i++) {
                assertEquals(imagePad, (int) tokens.get(starts[s] + i));
                // The fake tower numbers its rows per image: row i is filled with i + 1.
                assertEquals(i + 1f, plan.embeddingAt(starts[s] + i)[0], 0f);
            }
        }
        assertTrue(starts[0] < starts[1]);
        // Text positions keep the embedding lookup.
        assertNull(plan.embeddingAt(0));
        assertNull(plan.embeddingAt(tokens.size() - 1));
        assertEquals(tokens.size(), plan.promptLength());
    }

    // ---- MRoPE positions, from first principles ------------------------------------------------

    @Test
    public void planPositionsFollowMropeRulesIncludingDecode() {
        // Tokens: two text, one 2x2 merged image (grid 1,4,4 at merge 2), one text.
        float[][] rows = new float[4][2];
        for (int i = 0; i < 4; i++) rows[i][0] = i + 1;
        MultimodalPromptPlan plan =
                MultimodalPromptPlan.of(
                        new int[] {1, 2, 99, 99, 99, 99, 3},
                        List.of(new MultimodalPromptPlan.ImageSpan(2, new ImageGrid(1, 4, 4), rows)),
                        99,
                        2);
        assertArrayEquals(new int[] {0, 0, 0}, plan.ropeAt(0));
        assertArrayEquals(new int[] {1, 1, 1}, plan.ropeAt(1));
        // Merged-grid order is rows outer, columns inner: (t, h, w) from the span's start.
        assertArrayEquals(new int[] {2, 2, 2}, plan.ropeAt(2));
        assertArrayEquals(new int[] {2, 2, 3}, plan.ropeAt(3));
        assertArrayEquals(new int[] {2, 3, 2}, plan.ropeAt(4));
        assertArrayEquals(new int[] {2, 3, 3}, plan.ropeAt(5));
        // After the image the running position advances by max(h, w) / merge = 2, not by 4.
        assertArrayEquals(new int[] {4, 4, 4}, plan.ropeAt(6));
        assertEquals(-2, plan.positionDelta());
        // Decode continues from the prompt's largest position + 1.
        assertArrayEquals(new int[] {5, 5, 5}, plan.ropeAt(7));
        assertArrayEquals(new int[] {6, 6, 6}, plan.ropeAt(8));
        assertNull(plan.embeddingAt(7));
        // Rows handed out are copies; the staged row cannot be edited from outside.
        plan.embeddingAt(2)[0] = -1f;
        assertEquals(1f, plan.embeddingAt(2)[0], 0f);
    }

    @Test
    public void planRejectsSpansThatDisagreeWithTheTokens() {
        float[][] rows = new float[4][2];
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        MultimodalPromptPlan.of(
                                new int[] {1, 99, 99, 99, 99},
                                List.of(new MultimodalPromptPlan.ImageSpan(0, new ImageGrid(1, 4, 4), rows)),
                                99,
                                2));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        MultimodalPromptPlan.of(
                                new int[] {99, 99, 99, 99},
                                List.of(
                                        new MultimodalPromptPlan.ImageSpan(
                                                0, new ImageGrid(1, 4, 4), new float[3][2])),
                                99,
                                2));
    }

    // ---- text-only path is untouched ------------------------------------------------------------

    @Test
    public void textOnlyConversationsEncodeExactlyAsBeforeWithNoPlan() {
        var encoded =
                encoder.encodeConversation(
                        List.of(ChatMessage.of(ChatRole.USER, "What is the capital of France?")),
                        List.of());
        assertFalse(encoded.hasImages());
        assertNull(encoded.imagePlan());
        assertEquals(
                "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                tokenizer.decode(encoded.tokens()));
        assertEquals(0, model.preprocessCalls);
        assertEquals(0, model.towerCalls);
    }

    @Test
    public void theTokensOnlyEntryPointRefusesImages() throws IOException {
        var withImage =
                List.of(
                        new ChatMessage(
                                ChatRole.USER, List.of(new ChatContent.Image(png(64, 64)))));
        assertThrows(
                UnsupportedOperationException.class, () -> encoder.encode(withImage, List.of()));
    }

    // ---- content kind and capability --------------------------------------------------------------

    @Test
    public void imageContentIsValidatedCopiedAndRoleChecked() {
        assertThrows(IllegalArgumentException.class, () -> new ChatContent.Image(null));
        assertThrows(IllegalArgumentException.class, () -> new ChatContent.Image(new byte[0]));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ChatContent.Image(new byte[ChatContent.Image.MAX_ENCODED_BYTES + 1]));

        byte[] source = {1, 2, 3};
        ChatContent.Image image = new ChatContent.Image(source);
        source[0] = 9;
        assertEquals(1, image.encodedBytes()[0]);
        image.encodedBytes()[0] = 9;
        assertEquals(1, image.encodedBytes()[0]);

        new ChatMessage(ChatRole.USER, List.of(image, new ChatContent.Text("hi")));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ChatMessage(ChatRole.SYSTEM, List.of(image)));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ChatMessage(ChatRole.ASSISTANT, List.of(image)));
    }

    @Test
    public void imageCapabilityIsDerivedFromTheFormat() {
        assertTrue(DelegatingModel.capabilitiesOf(format).imageInput());
        assertFalse(DelegatingModel.capabilitiesOf(null).imageInput());
        assertFalse(ModelCapabilities.NONE.imageInput());
    }

    // ---- fixtures -----------------------------------------------------------------------------------

    private static byte[] png(int width, int height) throws IOException {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.setRGB(x, y, (x * 255 / width) << 16 | (y * 255 / height) << 8 | 0x40);
            }
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(image, "png", out);
        return out.toByteArray();
    }

    /** Real preprocessing, numbered fake tower rows; everything else is not the encoder's business. */
    private static final class FakeVisionModel implements Model, VisionModel {
        private final ChatFormat chatFormat;
        private final FlashNextVisionConfig config;
        private final FlashNextImagePreprocessor preprocessor;
        int preprocessCalls;
        int towerCalls;

        FakeVisionModel(ChatFormat chatFormat, FlashNextVisionConfig config) {
            this.chatFormat = chatFormat;
            this.config = config;
            this.preprocessor = new FlashNextImagePreprocessor(config);
        }

        @Override public FlashNextVisionConfig visionConfig() { return config; }

        @Override public PreprocessedImage preprocessImage(byte[] encodedBytes) throws IOException {
            preprocessCalls++;
            return preprocessor.preprocess(encodedBytes);
        }

        @Override public float[][] imageEmbeddings(PreprocessedImage image) {
            towerCalls++;
            float[][] rows = new float[config.tokenCount(image.grid())][config.outHiddenSize()];
            for (int i = 0; i < rows.length; i++) java.util.Arrays.fill(rows[i], i + 1f);
            return rows;
        }

        @Override public ChatFormat chatFormat() { return chatFormat; }
        @Override public boolean shouldAddBeginOfText() { return false; }
        @Override public ModelType getModelType() { return ModelType.QWEN_4_EXP; }
        @Override public Configuration configuration() { throw notNeeded(); }
        @Override public Tokenizer tokenizer() { throw notNeeded(); }
        @Override public Weights weights() { throw notNeeded(); }
        @Override public State createNewState() { throw notNeeded(); }
        @Override public State createNewState(int batchsize) { throw notNeeded(); }
        @Override public List<Integer> generateTokens(State state, int startPosition,
                List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                boolean echo, IntConsumer onTokenGenerated) { throw notNeeded(); }
        @Override public List<Integer> generateTokensGPU(State state, int startPosition,
                List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                boolean echo, IntConsumer onTokenGenerated, TornadoVMMasterPlan plan) { throw notNeeded(); }

        private static UnsupportedOperationException notNeeded() {
            return new UnsupportedOperationException("the encoder only reads the format and tower");
        }
    }

    /** Jinja2 rendering of {@code [{image}, {text}]} with {@code enable_thinking=False}. */
    private static final String EXPECTED_IMAGE_THEN_TEXT_DISABLED =
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    + "What is in this picture?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}
