package net.virtualvoid.llama2

import java.io.File

object Llama2Main extends App {
  val baseDir = { // reStart runs with a subdirectory as the working directory
    val firstTry = new File("tokenizer.bin")
    if (firstTry.exists()) firstTry.getParentFile
    else new File("..")
  }
  //val checkpointFile = new File(baseDir, "llama2_7b.bin")
  val checkpointFile = new File(baseDir, "stories15M.bin")
  //val checkpointFile = new File(baseDir, "stories42M.bin")
  //val checkpointFile = new File(baseDir, "stories110M.bin")
  val tokenizerFile = new File(baseDir, "tokenizer.bin")

  val ggmlFile = new File(baseDir, "llama-2-7b.ggmlv3.q4_0.bin")

  //val model = Llama2Model.fromLlama2CModel(checkpointFile, tokenizerFile)
  val model = Llama2Model.fromGgml(ggmlFile)

  val useTensor = true
  val transformer: Llama2Transformer =
    if (useTensor)
      Llama2TensorTransformer.init(model)
    else
      Llama2SimpleTransformer.init(model)

  VectMult.setParallelism(6)

  val sampler = TemperatureSampling(0.9f)

  def run(): Unit = {
    val steps = 30
    val runner = new Llama2Runner(transformer, model)
    val start = System.nanoTime()
    runner.iterate(steps, sampler = sampler).foreach { x => print(x); Console.out.flush() }

    println()
    val end = System.nanoTime()
    val lastedNanos = end - start
    val tokensPerSecond = steps.toFloat / lastedNanos * 1e9
    println(f"$tokensPerSecond%5.2f tokens per second")
  }
  while (true) run()
  run()
  run()
  run()
  run()
  run()
  run()
}