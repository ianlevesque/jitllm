package org.beehive.jitllm.tokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.*;
import org.beehive.jitllm.format.safetensors.MetadataJson;

/** Hugging Face byte-level BPE with explicit merge ranks and checkpoint added tokens. */
public final class Qwen4Tokenizer extends Qwen35Tokenizer {
    private record Merge(int rank, int token) {}
    private record Data(Map<String,Object> metadata, Vocabulary vocabulary, Map<Long,Merge> merges,
                        Map<String,Integer> special, Pattern pattern, boolean nfc) {}
    private final Data data;
    private Qwen4Tokenizer(Data data) {
        super(data.metadata(), data.vocabulary());
        this.data = data;
    }

    public static Qwen4Tokenizer load(Path directory) throws IOException {
        Map<String,Object> root = MetadataJson.parseObject(Files.readString(directory.resolve("tokenizer.json")));
        Map<String,Object> model = object(root.get("model"));
        boolean nfc = root.get("normalizer") != null && "NFC".equals(object(root.get("normalizer")).get("type"));
        if (!"BPE".equals(model.get("type")) || (root.get("normalizer") != null && !nfc)) throw new IOException("Unsupported Qwen4 tokenizer pipeline");
        if(model.get("dropout")!=null || model.get("unk_token")!=null || Boolean.TRUE.equals(model.get("ignore_merges"))
                || Boolean.TRUE.equals(model.get("byte_fallback")) || Boolean.TRUE.equals(model.get("fuse_unk"))
                || !"".equals(model.get("continuing_subword_prefix")) || !"".equals(model.get("end_of_word_suffix")))
            throw new IOException("Unsupported Qwen4 BPE options");
        Map<String,Object> vocab = object(model.get("vocab"));
        List<?> added = (List<?>) root.get("added_tokens");
        int size = vocab.size();
        for (Object token : added) size = Math.max(size, ((Number)object(token).get("id")).intValue() + 1);
        String[] tokens = new String[size];
        int[] types = new int[size];
        Arrays.fill(types,1);
        for (var entry : vocab.entrySet()) tokens[((Number) entry.getValue()).intValue()] = entry.getKey();
        Map<String,Integer> specials = new LinkedHashMap<>();
        for (Object value : added) {
            Map<String,Object> token = object(value);
            if (Boolean.TRUE.equals(token.get("lstrip")) || Boolean.TRUE.equals(token.get("rstrip"))
                    || Boolean.TRUE.equals(token.get("single_word")) || Boolean.TRUE.equals(token.get("normalized")))
                throw new IOException("Unsupported added-token matching options");
            int id = ((Number) token.get("id")).intValue();
            String text = (String) token.get("content");
            tokens[id] = text;
            types[id] = text.equals("<think>") || text.equals("</think>") ? 4 : 3;
            specials.put(text,id);
        }
        for (String token : tokens) if (token == null) throw new IOException("Tokenizer vocabulary has holes");
        Vocabulary vocabulary = new Vocabulary(tokens,null);
        List<?> rawMerges = (List<?>) model.get("merges");
        String[] lines = new String[rawMerges.size()];
        Map<Long,Merge> merges = new HashMap<>();
        for (int rank=0; rank<rawMerges.size(); rank++) {
            Object raw = rawMerges.get(rank);
            String[] pair = raw instanceof String text ? text.split(" ",2)
                    : ((List<?>)raw).stream().map(Object::toString).toArray(String[]::new);
            if (pair.length != 2) throw new IOException("Invalid BPE merge");
            int left = vocabulary.getIndex(pair[0]).orElseThrow(), right = vocabulary.getIndex(pair[1]).orElseThrow();
            int result = vocabulary.getIndex(pair[0]+pair[1]).orElseThrow();
            merges.putIfAbsent(key(left,right),new Merge(rank,result));
            lines[rank] = pair[0] + " " + pair[1];
        }
        Map<String,Object> pre = object(root.get("pre_tokenizer"));
        if (!"Sequence".equals(pre.get("type"))) throw new IOException("Expected Qwen4 tokenizer sequence");
        List<?> stages = (List<?>) pre.get("pretokenizers");
        if (stages.size()!=2) throw new IOException("Unexpected tokenizer stages");
        Map<String,Object> split = object(stages.get(0)), bytes = object(stages.get(1));
        if (!"Split".equals(split.get("type")) || !"Isolated".equals(split.get("behavior")) || Boolean.TRUE.equals(split.get("invert"))
                || !"ByteLevel".equals(bytes.get("type")) || Boolean.TRUE.equals(bytes.get("add_prefix_space")) || Boolean.TRUE.equals(bytes.get("use_regex")))
            throw new IOException("Unsupported byte-level pretokenizer");
        String regex = (String) object(split.get("pattern")).get("Regex");
        return new Qwen4Tokenizer(new Data(Map.of("tokenizer.ggml.token_type",types,"tokenizer.ggml.merges",lines),
                vocabulary, Map.copyOf(merges), Map.copyOf(specials), Pattern.compile(regex), nfc));
    }

    private static long key(int left,int right) { return ((long)left<<32) | (right & 0xffffffffL); }
    @SuppressWarnings("unchecked") private static Map<String,Object> object(Object value) throws IOException {
        if (!(value instanceof Map)) throw new IOException("Expected tokenizer object");
        return (Map<String,Object>)value;
    }
    @Override public String regexPattern() { return data.pattern().pattern(); }
    @Override public Map<String,Integer> getSpecialTokens() { return data.special(); }
    @Override public boolean isSpecialToken(int id) { return data.special().containsValue(id) && !shouldDisplayToken(id); }
    @Override public boolean shouldDisplayToken(int id) {
        return id>=0 && id<data.vocabulary().size() && super.shouldDisplayToken(id);
    }
    public int tokenCount() { return data.vocabulary().size(); }

    @Override public List<Integer> encodeOrdinary(String text) {
        if(data.nfc()) text=Normalizer.normalize(text,Normalizer.Form.NFC);
        List<Integer> result = new ArrayList<>();
        Matcher chunks = data.pattern().matcher(text);
        while(chunks.find()) {
            List<Integer> ids = new ArrayList<>();
            for(byte b: chunks.group().getBytes(StandardCharsets.UTF_8))
                ids.add(data.vocabulary().getIndex(new String(Character.toChars(BYTE_ENCODER.get(b & 255)))).orElseThrow());
            while(ids.size()>1) {
                int best=-1, rank=Integer.MAX_VALUE, token=-1;
                for(int i=0; i+1<ids.size(); i++) {
                    Merge merge=data.merges().get(key(ids.get(i),ids.get(i+1)));
                    if(merge!=null && merge.rank()<rank) { best=i; rank=merge.rank(); token=merge.token(); }
                }
                if(best<0) break;
                ids.set(best,token);
                ids.remove(best+1);
            }
            result.addAll(ids);
        }
        return result;
    }

    @Override public List<Integer> encode(String text, Set<String> allowedSpecial) {
        if (!data.special().keySet().containsAll(allowedSpecial)) throw new IllegalArgumentException("Unknown allowed special token");
        if(allowedSpecial.isEmpty()) return encodeOrdinary(text);
        String pattern=allowedSpecial.stream().sorted(Comparator.comparingInt(String::length).reversed()).map(Pattern::quote)
                .collect(java.util.stream.Collectors.joining("|"));
        Matcher matcher=Pattern.compile(pattern).matcher(text);
        List<Integer> result=new ArrayList<>();
        int end=0;
        while(matcher.find()) {
            result.addAll(encodeOrdinary(text.substring(end,matcher.start())));
            result.add(data.special().get(matcher.group()));
            end=matcher.end();
        }
        result.addAll(encodeOrdinary(text.substring(end)));
        return result;
    }
}
