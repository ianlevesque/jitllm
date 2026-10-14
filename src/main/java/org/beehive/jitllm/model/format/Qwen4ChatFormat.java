package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.tokenizer.Qwen4Tokenizer;

/** Flash-Next ChatML and XML tools, rendered as the checkpoint's chat template renders them. */
public final class Qwen4ChatFormat extends Qwen35ChatFormat {
    public Qwen4ChatFormat(Qwen4Tokenizer tokenizer) {
        super(tokenizer, new ChatTokens("<|im_start|>", "<|im_end|>", "", "<|endoftext|>", ""));
    }

    /** The checkpoint template trims every message's rendered content ({@code | trim}). */
    @Override public List<Integer> encodeMessage(Message message) {
        var tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeOrdinaryAsList(message.content().strip()));
        if (!List.of(Role.FIM_PREFIX, Role.FIM_SUFFIX, Role.FIM_MIDDLE).contains(message.role())) {
            tokens.addAll(endOfTurn());
        }
        return tokens;
    }

    /** Tool-result content is trimmed by that same template {@code | trim}; the base renderer keeps it verbatim. */
    @Override public List<Integer> encodeToolResults(List<ToolResult> results) {
        return super.encodeToolResults(results.stream()
                .map(r -> r.content() == null ? r
                        : new ToolResult(r.toolCallId(), r.toolName(), r.content().strip()))
                .toList());
    }

    @Override public boolean supportsImages() {
        return true;
    }

    /**
     * A mixed text/image turn as the checkpoint template plus its image processor render it:
     * the template writes {@code <|vision_start|><|image_pad|><|vision_end|>} per image item,
     * and the processor expands the single placeholder to the image's merged token count. The
     * rendered content is trimmed as a whole ({@code | trim}), which touches only whitespace at
     * the two ends of the concatenation — never whitespace between an image and its neighbour.
     */
    @Override public List<Integer> encodeRichMessage(Role role, List<ContentPart> parts) {
        var tokens = encodeHeader(new Message(role, ""));
        for (ContentPart part : trimParts(parts)) {
            switch (part) {
                case ContentPart.Text text -> tokens.addAll(tokenizer.encodeOrdinaryAsList(text.text()));
                case ContentPart.Image image -> {
                    tokens.add(visionToken("<|vision_start|>"));
                    int imagePad = visionToken("<|image_pad|>");
                    for (int i = 0; i < image.tokenCount(); i++) tokens.add(imagePad);
                    tokens.add(visionToken("<|vision_end|>"));
                }
            }
        }
        tokens.addAll(endOfTurn());
        return tokens;
    }

    private int visionToken(String spelling) {
        Integer token = tokenizer.getSpecialTokens().get(spelling);
        if (token == null) throw new IllegalStateException("Tokenizer has no " + spelling + " token");
        return token;
    }

    /** The template's whole-content {@code | trim}, applied across the leading/trailing text parts. */
    private static List<ContentPart> trimParts(List<ContentPart> parts) {
        List<ContentPart> trimmed = new ArrayList<>(parts);
        while (!trimmed.isEmpty() && trimmed.getFirst() instanceof ContentPart.Text first) {
            String stripped = first.text().stripLeading();
            if (!stripped.isEmpty()) {
                trimmed.set(0, new ContentPart.Text(stripped));
                break;
            }
            trimmed.removeFirst();
        }
        while (!trimmed.isEmpty() && trimmed.getLast() instanceof ContentPart.Text last) {
            String stripped = last.text().stripTrailing();
            if (!stripped.isEmpty()) {
                trimmed.set(trimmed.size() - 1, new ContentPart.Text(stripped));
                break;
            }
            trimmed.removeLast();
        }
        return trimmed;
    }
}
