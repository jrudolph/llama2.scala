package net.virtualvoid.llama2

import spray.json.*
import DefaultJsonProtocol.*

import java.io.{ File, FileOutputStream, PrintWriter }
import scala.io.Source

object BenchmarkModel {
  final case class PrimaryMetric(
      rawData:          Seq[Seq[Float]],
      score:            Float,
      scoreConfidence:  Seq[Float],
      scoreError:       Float,
      scorePercentiles: Map[String, Float],
      scoreUnit:        String
  )

  implicit val PrimaryMetricFormat: JsonFormat[PrimaryMetric] = jsonFormat6(PrimaryMetric.apply _)
  final case class SecondaryMetrics()

  implicit val SecondaryMetricsFormat: JsonFormat[SecondaryMetrics] = jsonFormat0(SecondaryMetrics.apply _)

  final case class BenchmarkRun(
      benchmark:             String,
      forks:                 Int,
      jdkVersion:            String,
      jmhVersion:            String,
      jvm:                   String,
      jvmArgs:               Seq[String],
      measurementBatchSize:  Int,
      measurementIterations: Int,
      measurementTime:       String,
      mode:                  String,
      params:                Map[String, String],
      primaryMetric:         PrimaryMetric,
      secondaryMetrics:      SecondaryMetrics,
      threads:               Int,
      vmName:                String,
      vmVersion:             String,
      warmupBatchSize:       Int,
      warmupIterations:      Int,
      warmupTime:            String
  )

  implicit val RootArrayElementFormat: JsonFormat[BenchmarkRun] = jsonFormat19(BenchmarkRun.apply _)
}

/*
| stories42M.bin             | none         | llama2.c / run     | 1       | 21.2    |
| stories42M.bin             | none         | llama2.c / runfast | 1       | 68.8    |
| stories42M.bin             | none         | llama2.c / runomp  | 1       | 98      |
| stories42M.bin             | none         | llama2.c / runomp  | 6       | 195     |
 */

object GenerateBenchmarkMarkdownTable extends App {
  val benchFile = new File("bench-full-2023-08-09-15-30.json")
  val out = new File("bench.md")

  val runs = Source.fromFile(benchFile).mkString.parseJson.convertTo[Seq[BenchmarkModel.BenchmarkRun]]

  val w = new PrintWriter(new FileOutputStream(out))
  import w._

  println("| Model | Quantization | Implementation | Threads | Score |")
  println("| ----- | ------------ | -------------- | ------- | ----- |")

  runs.sortBy(r => (r.params("model"), r.params("quantization"), r.params("impl"), r.params("threads").toInt)).foreach { run =>
    val params = run.params
    val model = params("model")
    val quantization = params("quantization")
    val implementation = params("impl")
    val threads = params("threads")
    val score = run.primaryMetric.score.formatted("%5.3f")

    println(s"| $model | $quantization | $implementation | $threads | $score |")
  }
  w.close()
}
