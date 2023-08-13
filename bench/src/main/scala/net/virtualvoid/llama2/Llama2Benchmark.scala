package net.virtualvoid.llama2

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.io.File

@State(Scope.Thread)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(
  value = 3,
  jvmArgs = Array(
    "-server",
    "-Xms15g",
    "-Xmx15g",
    "-XX:NewSize=500m",
    "-XX:MaxNewSize=500m",
    "-XX:InitialCodeCacheSize=512m",
    "-XX:ReservedCodeCacheSize=512m",
    "-XX:+AlwaysPreTouch"))
@BenchmarkMode(Array(Mode.Throughput))
@OutputTimeUnit(TimeUnit.SECONDS)
abstract class CommonBenchmark

object BenchmarkConfig {
  val Steps = 20
}

class Llama2Benchmark extends CommonBenchmark {
  @Param(Array("stories15M.bin", "stories42M.bin", "stories110M.bin", "llama2_7b.bin", "llama-2-7b.ggmlv3.q4_0.bin"))
  var model: String = _

  @Param(Array("scala", "native-avx2"))
  var impl: String = _

  @Param(Array("1", "2", "4", "6", "8", "10", "12", "14", "16"))
  var threads: Int = _

  @Param(Array("none", "Q8", "Q4"))
  var quantization: String = _

  var _model: Llama2Model = _
  var _impl: MathImplementation = _
  var _transformer: Llama2TensorTransformer = _

  @Setup
  def setup(): Unit = {
    if (model == "llama2_7b.bin" && quantization == "none")
      throw new IllegalArgumentException("llama2_7b.bin is too large to run without quantization")

    val baseDir = { // reStart runs with a subdirectory as the working directory
      val firstTry = new File("tokenizer.bin")
      if (firstTry.exists()) firstTry.getParentFile
      else new File("..")
    }
    def f(name: String): File = new File(baseDir, name)

    val quantize: Llama2Model => Llama2Model = quantization match {
      case "none" => identity
      case "Q4"   => _.quantizeQ4
      case "Q8"   => _.quantizeQ8
    }

    _impl = impl match {
      case "scala" =>
        if (threads != 1) throw new IllegalArgumentException("Scala implementation does not support parallelism")
        ScalaMathImplementation
      case "native-avx2" => AVX2MathImplementation
    }
    val model0 =
      if (model.contains("ggml"))
        Llama2Model.fromGgml(f(model))
      else
        Llama2Model.fromLlama2CModel(f(model), f("tokenizer.bin"))
    _model = quantize(model0)

    MathImplementation.setImplementation(_impl)
    VectMult.setParallelism(threads)
    _transformer = Llama2TensorTransformer.init(_model)
  }

  @Benchmark
  @OperationsPerInvocation(5 /*BenchmarkConfig.Steps*/ )
  def run(blackhole: Blackhole): Unit = {
    val runner = Llama2Runner(_transformer, _model)
    val it: Iterator[String] = runner.iterate(5 /*BenchmarkConfig.Steps*/ )
    val result: String = it.mkString

    blackhole.consume(result)
  }
}
