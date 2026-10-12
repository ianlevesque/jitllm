package org.beehive.jitllm.model.qwen4;

import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.io.ByteArrayOutputStream;
import org.beehive.jitllm.backend.cpu.Qwen4QsaAttention;
import org.beehive.jitllm.backend.cpu.Qwen4DeltaNet;
import org.beehive.jitllm.format.safetensors.*;
import org.junit.Test;
import static org.junit.Assert.*;

public class Qwen4PrefillPrecisionTest {
    private static final class Fixture implements AutoCloseable {
        final Path directory=Files.createTempDirectory("qsa-prefill-precision-");
        final PagedTensorReader reader;
        final Qwen4QsaAttention layer;
        final Qwen4DeltaNet delta;
        Fixture() throws Exception {
            Files.writeString(directory.resolve("config.json"),Qwen4ConfigurationTest.CONFIG);
            Map<String,Object> header=new LinkedHashMap<>();var data=new ByteArrayOutputStream();
            float[][] q=new float[32][8];for(int h=0;h<2;h++) for(int i=0;i<8;i++) q[h*16+i][i]=1;
            add(header,data,"q_proj.weight",q);
            for(String name:List.of("k_proj.weight","v_proj.weight")) {
                float[][] matrix=new float[8][8];for(int i=0;i<8;i++) matrix[i][i]=1;add(header,data,name,matrix);
            }
            float[][] output=new float[8][16];for(int i=0;i<8;i++) output[i][i]=1;
            add(header,data,"o_proj.weight",output);add(header,data,"indexer.index_qk_proj.weight",new float[12][8]);
            add(header,data,"q_norm.weight",new float[1][8]);add(header,data,"k_norm.weight",new float[1][8]);
            add(header,data,"indexer.q_layernorm.weight",new float[1][4]);add(header,data,"indexer.k_layernorm.weight",new float[1][4]);
            float[][] qkv=new float[32][8];for(var row:qkv) Arrays.fill(row,1);
            add(header,data,"gdn.in_proj_qkv.weight",qkv);add(header,data,"gdn.in_proj_z.weight",new float[16][8]);
            add(header,data,"gdn.in_proj_a.weight",new float[4][8]);add(header,data,"gdn.in_proj_b.weight",new float[4][8]);
            float[][] down=new float[8][16];for(int i=0;i<8;i++) down[i][i]=1;add(header,data,"gdn.out_proj.weight",down);
            float[][] conv=new float[32][4];for(var row:conv) row[3]=1;add(header,data,"gdn.conv1d.weight",conv);
            add(header,data,"gdn.norm.weight",new float[][]{{1,1,1,1}});
            add(header,data,"gdn.A_log",new float[1][4]);add(header,data,"gdn.dt_bias",new float[1][4]);
            byte[] metadata=MetadataJson.write(header).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            var bytes=ByteBuffer.allocate(8+metadata.length+data.size()).order(ByteOrder.LITTLE_ENDIAN);
            bytes.putLong(metadata.length).put(metadata).put(data.toByteArray());Files.write(directory.resolve("model.safetensors"),bytes.array());
            reader=new PagedTensorReader(4096,64);
            layer=new Qwen4QsaAttention(Qwen4Configuration.load(directory,-1),SafeTensorIndex.open(directory),reader,"test",true,true);
            delta=new Qwen4DeltaNet(Qwen4Configuration.load(directory,-1),SafeTensorIndex.open(directory),reader,"test.gdn");
        }
        private static void add(Map<String,Object> header,ByteArrayOutputStream data,String name,float[][] values) throws Exception {
            int start=data.size();var bytes=ByteBuffer.allocate(values.length*values[0].length*2).order(ByteOrder.LITTLE_ENDIAN);
            for(var row:values) for(float value:row) bytes.putShort((short)(Float.floatToRawIntBits(value)>>>16));
            data.write(bytes.array());
            header.put("test."+name,Map.of("dtype","BF16","shape",List.of(values.length,values[0].length),"data_offsets",List.of(start,data.size())));
        }
        public void close() throws Exception {
            reader.close();try(var files=Files.walk(directory)) {for(var file:files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);}
        }
    }
    private static float[] input() {float[] input=new float[8];Arrays.fill(input,1.03125f);return input;}
    @Test public void freshPrefillReadsBfloatValuesWhileDecodeReadsTheFp8Cache() throws Exception {
        try(var fixture=new Fixture()) {
            var fresh=fixture.layer.forward(input(),fixture.layer.newState(),true);
            var decoded=fixture.layer.forward(input(),fixture.layer.newState());
            assertEquals(.515625f,fresh[0],0);assertEquals(.5f,decoded[0],0);
        }
    }
    @Test public void rollbackAcrossFreshViewReleaseRestoresPrefillExactly() throws Exception {
        try(var fixture=new Fixture()) {
            var state=fixture.layer.newState();fixture.layer.forward(input(),state,true);
            var before=state.snapshot();var expected=fixture.layer.forward(input(),state,true);
            state.finishFreshPrefill();state.restore(before);
            assertArrayEquals(expected,fixture.layer.forward(input(),state,true),0f);
        }
    }
    @Test public void zeroRecurrentOutputGateUsesSigmoidHalfRatherThanSiluZero() throws Exception {
        try(var fixture=new Fixture()) {
            float[] input=new float[8];Arrays.fill(input,1);
            var output=fixture.delta.forward(input,fixture.delta.newState());
            for(float value:output) assertEquals(.5f,value,.001f);
        }
    }
    @Test public void recurrentCheckpointIsOwnedByOneSequenceAndCloseRejectsForward() throws Exception {
        try(var fixture=new Fixture();var first=fixture.delta.newState();var second=fixture.delta.newState();var saved=first.snapshot()) {
            assertThrows(IllegalArgumentException.class,()->second.restore(saved));
            first.close();assertThrows(IllegalStateException.class,()->fixture.delta.forward(input(),first));
            assertThrows(IllegalStateException.class,first::snapshot);
        }
    }
    @Test public void attentionCheckpointIsOwnedByOneSequenceAndCloseRejectsForward() throws Exception {
        try(var fixture=new Fixture();var first=fixture.layer.newState();var second=fixture.layer.newState();var saved=first.snapshot()) {
            assertThrows(IllegalArgumentException.class,()->second.restore(saved));
            first.close();assertThrows(IllegalStateException.class,()->fixture.layer.forward(input(),first));
            assertThrows(IllegalStateException.class,first::snapshot);
            assertThrows(IllegalStateException.class,first::selectedTokens);
        }
    }
}
