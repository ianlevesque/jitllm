package org.beehive.jitllm.integration.cli;

import java.nio.file.*;
import java.util.*;
import org.beehive.jitllm.backend.tornado.FlashNextMatrixExecutor;
import org.beehive.jitllm.format.safetensors.*;

/** Check actual GPU graph reuse, W4A4, weight eviction and private mapping safety. */
public final class FlashNextCacheProbe {
    public static void main(String[] args) throws Exception {
        if(args.length!=3) throw new IllegalArgumentException("Usage: FlashNextCacheProbe MODEL_DIR REQUEST.json DEVICE_INDEX");
        var request=MetadataJson.parseObject(Files.readString(Path.of(args[1])));
        var prefixes=((List<?>)request.get("matrices")).stream().map(Object::toString).toList();
        var index=SafeTensorIndex.open(Path.of(args[0]),name->prefixes.stream().anyMatch(prefix->name.startsWith(prefix+".")));
        List<Object> comparisons=new ArrayList<>();
        try(var reader=new PagedTensorReader(1048576,4096)) {
            var reference=new ArrayList<ModelOptMatrix>();
            var gpu=new ArrayList<ModelOptMatrix>();
            long largest=0;
            for(var prefix:prefixes) {
                var matrix=new ModelOptMatrix(index,reader,prefix);reference.add(matrix);
                largest=Math.max(largest,matrix.weights().bytes()+(matrix.blockScales()==null?0:matrix.blockScales().bytes())+4L*(matrix.rows()+matrix.columns()+1024));
            }
            long budget=((Number)request.getOrDefault("cache_bytes",largest+65536)).longValue();
            try(var executor=new FlashNextMatrixExecutor(Integer.parseInt(args[2]),budget)) {
                for(var prefix:prefixes) gpu.add(new ModelOptMatrix(index,reader,prefix,executor));
                for(int round=0;round<3;round++) for(int number=0;number<gpu.size();number++) {
                    var matrix=reference.get(number);float[] input=new float[matrix.columns()],expected=new float[matrix.rows()],actual=new float[matrix.rows()];
                    for(int i=0;i<input.length;i++) input[i]=FloatFormats.roundBf16((float)Math.sin(i*.017+round*.23));
                    long beforeUploads=executor.statistics().uploads();
                    var tensor=matrix.weights();
                    byte[] before;
                    try(var channel=java.nio.channels.FileChannel.open(tensor.file())) {
                        var buffer=java.nio.ByteBuffer.allocate(32);SafeTensorIndex.readFully(channel,buffer,tensor.offset()-32);before=buffer.array();
                    }
                    for(int mode=0;mode<(matrix.dtype().equals("U8")?3:2);mode++) {
                        // Alternate plain/packed/plain on the same graph to catch stale alpha
                        // and input-transfer bugs. The second call must reuse the weights.
                        if(mode==1 && matrix.dtype().equals("U8")) {
                            var packed=Nvfp4Activation.quantize(input,.002f);
                            matrix.multiplyNvfp4(packed,expected);gpu.get(number).multiplyNvfp4(packed,actual);
                        } else {matrix.multiplyReference(input,expected);gpu.get(number).multiply(input,actual);}
                        double largestError=0;
                        for(int i=0;i<actual.length;i++) {
                            double error=Math.abs(actual[i]-expected[i]);largestError=Math.max(largestError,error);
                            if(!Float.isFinite(actual[i]) || error>.002+.0002*Math.abs(expected[i])) throw new AssertionError("GPU cached projection mismatch at "+prefixes.get(number)+" row "+i);
                        }
                        comparisons.add(Map.of("matrix",prefixes.get(number),"round",round,"packed_activation",mode==1&&matrix.dtype().equals("U8"),"max_absolute_error",largestError));
                    }
                    if(executor.statistics().uploads()-beforeUploads>1) throw new AssertionError("Repeated projection reuploaded weights");
                    try(var channel=java.nio.channels.FileChannel.open(tensor.file())) {
                        var buffer=java.nio.ByteBuffer.allocate(32);SafeTensorIndex.readFully(channel,buffer,tensor.offset()-32);
                        if(!Arrays.equals(before,buffer.array())) throw new AssertionError("Checkpoint header-neighbor bytes changed");
                    }
                    if(executor.statistics().residentBytes()>budget) throw new AssertionError("GPU cache exceeded its budget");
                }
                var stats=executor.statistics();
                if(stats.kernelNanos()<=0 || (gpu.size()>1 && stats.evictions()==0)) throw new AssertionError("Missing GPU execution or cache eviction");
                System.out.println(MetadataJson.write(Map.of("device",executor.deviceName(),"cache_budget_bytes",budget,"cache_resident_bytes",stats.residentBytes(),
                        "uploads",stats.uploads(),"evictions",stats.evictions(),"executions",stats.executions(),"gpu_kernel_ns",stats.kernelNanos(),
                        "checkpoint_bytes_unchanged",true,"comparisons",comparisons)));
            }
        }
    }
}
