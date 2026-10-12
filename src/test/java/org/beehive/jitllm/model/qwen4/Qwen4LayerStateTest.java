package org.beehive.jitllm.model.qwen4;

import java.io.ByteArrayOutputStream;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.cpu.Qwen4CpuLayers;
import org.beehive.jitllm.format.safetensors.*;
import org.junit.Test;
import static org.junit.Assert.*;

public final class Qwen4LayerStateTest {
    private static final class Fixture implements AutoCloseable {
        final Path directory=Files.createTempDirectory("qwen4-hc-ple-");
        final Map<String,Object> header=new LinkedHashMap<>();
        final ByteArrayOutputStream data=new ByteArrayOutputStream();
        final PagedTensorReader reader;
        final Qwen4CpuLayers.HyperConnection hyper;
        final Qwen4CpuLayers.Ple ple;
        Fixture() throws Exception {
            Files.writeString(directory.resolve("config.json"),Qwen4ConfigurationTest.CONFIG);
            bf("test.hyper.hc_norm.weight",new float[1][32]);
            float[][] down=new float[2][32];down[0][0]=.5f;down[0][1]=.001953125f;
            bf("test.hyper.input_mix_weight_down.weight",down);
            float[][] up=new float[32][2];for(var row:up) row[0]=2;
            bf("test.hyper.input_mix_weight_up.weight",up);bf("test.hyper.block_inject_weight.weight",new float[4][32]);
            String table="test.ple.ple_embedding";
            longs(table+".layer_multipliers",3,5,7);longs(table+".ngram_heads_vocab_sizes",3,3,3,3);longs(table+".ngram_heads_offsets",0,3,6,9);
            bf(table+".ngram_embedding.weight_scale",new float[][]{{1}});
            byte[] rows=new byte[48];Arrays.fill(rows,(byte)0x30);add(table+".ngram_embedding.weight","F8_E4M3",List.of(12,4),rows);
            float[][] key=new float[32][16];for(int i=0;i<32;i++) key[i][i%16]=1;
            float[][] value=new float[8][16];for(int i=0;i<8;i++) value[i][i]=1;
            bf("test.ple.key_proj.weight",key);bf("test.ple.value_proj.weight",value);
            for(String name:List.of("norm_key","norm_query","norm_conv")) bf("test.ple."+name+".weight",new float[1][32]);
            float[][] conv=new float[32][4];for(var row:conv) row[3]=1;bf("test.ple.conv1d.weight",conv);
            byte[] metadata=MetadataJson.write(header).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            var file=ByteBuffer.allocate(8+metadata.length+data.size()).order(ByteOrder.LITTLE_ENDIAN);
            file.putLong(metadata.length).put(metadata).put(data.toByteArray());Files.write(directory.resolve("model.safetensors"),file.array());
            reader=new PagedTensorReader(4096,64);var index=SafeTensorIndex.open(directory);var config=Qwen4Configuration.load(directory,-1);
            hyper=new Qwen4CpuLayers.HyperConnection(config,index,reader,"test.hyper",true);
            ple=new Qwen4CpuLayers.Ple(config,index,reader,"test.ple");
        }
        void add(String name,String dtype,List<Integer> shape,byte[] bytes) throws Exception {
            int start=data.size();data.write(bytes);header.put(name,Map.of("dtype",dtype,"shape",shape,"data_offsets",List.of(start,data.size())));
        }
        void bf(String name,float[][] values) throws Exception {
            var bytes=ByteBuffer.allocate(values.length*values[0].length*2).order(ByteOrder.LITTLE_ENDIAN);
            for(var row:values) for(float value:row) bytes.putShort((short)(Float.floatToRawIntBits(value)>>>16));
            add(name,"BF16",List.of(values.length,values[0].length),bytes.array());
        }
        void longs(String name,long... values) throws Exception {
            var bytes=ByteBuffer.allocate(values.length*8).order(ByteOrder.LITTLE_ENDIAN);for(long value:values) bytes.putLong(value);
            add(name,"I64",List.of(values.length),bytes.array());
        }
        public void close() throws Exception {
            hyper.close();reader.close();try(var files=Files.walk(directory)) {for(var file:files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);}
        }
    }
    private static float[] hidden() {float[] values=new float[32];Arrays.fill(values,1);return values;}
    @Test public void seventeenthPrefillRowUsesTheBfloatProjectionBoundary() throws Exception {
        try(var fixture=new Fixture()) {
            for(float value:fixture.hyper.mix(hidden(),16).output()) assertEquals(.53515625f,value,0);
            for(float value:fixture.hyper.mix(hidden(),17).output()) assertEquals(.53125f,value,0);
            assertArrayEquals(fixture.hyper.mix(hidden(),17).output(),fixture.hyper.mix(hidden(),4096).output(),0);
        }
    }
    @Test public void pleCheckpointIsRequestOwnedAndCloseRejectsExecution() throws Exception {
        try(var fixture=new Fixture();var first=fixture.ple.newState();var second=fixture.ple.newState();var before=first.snapshot()) {
            assertThrows(IllegalArgumentException.class,()->second.restore(before));
            float[] expected=fixture.ple.forward(2,hidden(),first);first.restore(before);
            assertArrayEquals(expected,fixture.ple.forward(2,hidden(),first),0);
            first.reset();assertArrayEquals(expected,fixture.ple.forward(2,hidden(),first),0);
            first.close();assertThrows(IllegalStateException.class,()->fixture.ple.forward(2,hidden(),first));
            assertThrows(IllegalStateException.class,first::snapshot);
        }
    }
    @Test public void pleApplyRoundsTheDeltaBeforeAddingTheResidual() throws Exception {
        try(var fixture=new Fixture();var state=fixture.ple.newState();var before=state.snapshot()) {
            float[] delta=fixture.ple.forward(2,hidden(),state,true);state.restore(before);
            float[] applied=fixture.ple.apply(2,hidden(),state,true);
            for(int i=0;i<applied.length;i++) assertEquals(FloatFormats.roundBf16(1+delta[i]),applied[i],0);
            fixture.hyper.close();assertThrows(IllegalStateException.class,()->fixture.hyper.mix(hidden()));
        }
    }
}
