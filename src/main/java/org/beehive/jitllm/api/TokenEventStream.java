package org.beehive.jitllm.api;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.IntFunction;
import org.beehive.jitllm.inference.sampler.SampledTokenLogprob;
import org.beehive.jitllm.tokenizer.Tokenizer;

/**
 * Turns the loop's raw token ids into the facade's ordered event stream [A5].
 *
 * <h2>Why a token is held back</h2>
 *
 * <p>The generation loop's internal callback fires <b>before</b> its stop check, so it sees the
 * terminal stop token. That is an accident of ordering rather than a contract, and publishing it
 * would put control-token ids in the public stream forever. So one token is held: when the next
 * arrives, the held one is emitted; at the end, a held terminal stop token is dropped and anything
 * else is emitted.
 *
 * <p>The cost is one token of latency, and it buys two guarantees a consumer can rely on: the
 * stream contains no control tokens, and <b>the number of events equals {@link
 * GenerationResult#generatedTokens()}</b>.
 *
 * <p>A stop token that appears mid-stream — possible only with the ignore-end-of-sequence debug
 * flag, which makes the loop continue past one — is emitted, because it is then genuinely part of
 * the output and is counted as such.
 *
 * <h2>Incremental decoding</h2>
 *
 * <p>A token can carry part of a multi-byte character, and decoding it alone yields a replacement
 * character rather than text. Such a token is held in a pending list and its event carries empty
 * text; when the bytes complete, the text is attached to the event that completes them.
 * Concatenating every non-empty text therefore equals the full response.
 */
final class TokenEventStream {

    private static final char REPLACEMENT = '�';

    private final Tokenizer tokenizer;
    private final Set<Integer> stopTokens;
    private final Consumer<GenerationEvent> onEvent;
    private final Consumer<String> onText;

    /**
     * Supplies the sampler's captured entry for a token id, in sample order, or {@code null}
     * when log-probabilities were not asked for. The sampler records an entry as it returns a
     * token, before the loop hands that token to {@link #accept(int)}, so the entry for a token
     * is already there when it arrives here; it is held with the token and dropped with a
     * terminal stop token, for the same reason the token is.
     */
    private final IntFunction<SampledTokenLogprob> logprobForToken;

    /** Tokens decoded so far that have not yet produced complete text. */
    private final List<Integer> pending = new ArrayList<>();

    private final StringBuilder text = new StringBuilder();

    private boolean holding;
    private int heldToken;
    private SampledTokenLogprob heldLogprob;

    TokenEventStream(
            Tokenizer tokenizer,
            Set<Integer> stopTokens,
            Consumer<GenerationEvent> onEvent,
            Consumer<String> onText) {
        this(tokenizer, stopTokens, onEvent, onText, null);
    }

    TokenEventStream(
            Tokenizer tokenizer,
            Set<Integer> stopTokens,
            Consumer<GenerationEvent> onEvent,
            Consumer<String> onText,
            IntFunction<SampledTokenLogprob> logprobForToken) {
        this.tokenizer = tokenizer;
        this.stopTokens = stopTokens;
        this.onEvent = onEvent;
        this.onText = onText;
        this.logprobForToken = logprobForToken;
    }

    /** Called for every token the loop produces, terminal stop token included. */
    void accept(int token) {
        if (holding) {
            emit(heldToken, heldLogprob);
        }
        heldToken = token;
        heldLogprob = logprobForToken == null ? null : logprobForToken.apply(token);
        holding = true;
    }

    /**
     * Generation is over. A held terminal stop token is dropped; anything else is emitted.
     *
     * @return the complete response text
     */
    String finish() {
        if (holding && !stopTokens.contains(heldToken)) {
            emit(heldToken, heldLogprob);
        }
        holding = false;
        heldLogprob = null;
        return text.toString();
    }

    private void emit(int token, SampledTokenLogprob sampled) {
        String piece = decodeIncrementally(token);
        text.append(piece);
        // onEvent first, then onToken for the same event [A5]. A throwing onEvent means
        // onToken does not see this token: the exception propagates, and no later callback for the
        // event runs.
        if (onEvent != null) {
            onEvent.accept(new GenerationEvent(token, piece, tokenLogprob(token, piece, sampled)));
        }
        if (onText != null && !piece.isEmpty()) {
            onText.accept(piece);
        }
    }

    /**
     * The facade entry for one emitted token, or {@code null} when the sampler supplied none —
     * not asked for, or an executing path that sampled somewhere the capture could not see (a
     * device-side sampler), where inventing a number would be worse than reporting none.
     * Alternative texts are single-token decodes; a token the tokenizer does not display keeps an
     * empty text, as an event for it would.
     */
    private TokenLogprob tokenLogprob(int token, String piece, SampledTokenLogprob sampled) {
        if (sampled == null || sampled.tokenId() != token) {
            return null;
        }
        List<TokenLogprob.Alternative> alternatives = new ArrayList<>(sampled.topLogprobs().size());
        for (SampledTokenLogprob.Top top : sampled.topLogprobs()) {
            String text =
                    tokenizer.shouldDisplayToken(top.tokenId())
                            ? tokenizer.decode(List.of(top.tokenId()))
                            : "";
            alternatives.add(new TokenLogprob.Alternative(top.tokenId(), text, top.logprob()));
        }
        return new TokenLogprob(token, piece, sampled.logprob(), alternatives);
    }

    /**
     * The text this token completed — empty while a character is still incomplete, or for a token
     * the tokenizer does not display.
     */
    private String decodeIncrementally(int token) {
        if (!tokenizer.shouldDisplayToken(token)) {
            return "";
        }
        pending.add(token);
        String decoded = tokenizer.decode(pending);
        if (decoded.indexOf(REPLACEMENT) >= 0) {
            // Still mid-character. The id is real; the text is not there yet.
            return "";
        }
        pending.clear();
        return decoded;
    }
}
