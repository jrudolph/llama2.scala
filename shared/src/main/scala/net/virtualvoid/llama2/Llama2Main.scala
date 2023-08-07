package net.virtualvoid.llama2

import java.io.File

object Llama2Main extends App {
  val baseDir = { // reStart runs with a subdirectory as the working directory
    val firstTry = new File("tokenizer.bin")
    if (firstTry.exists()) firstTry.getParentFile
    else new File("..")
  }
  //val checkpointFile = new File(baseDir, "llama2_7b.bin")
  //val checkpointFile = new File(baseDir, "stories15M.bin")
  //val checkpointFile = new File(baseDir, "stories42M.bin")
  val checkpointFile = new File(baseDir, "stories110M.bin")
  val tokenizerFile = new File(baseDir, "tokenizer.bin")

  val ggmlFile = new File(baseDir, "llama-2-7b.ggmlv3.q4_0.bin")

  val (config, vocab, weights) = GgmlLoader.fromGgml(ggmlFile)
  //val config = Config.fromFile(checkpointFile)
  //val vocab = Vocab.fromFile(config, tokenizerFile)
  //val weights = Weights.fromFile(config, checkpointFile)

  val useTensor = true
  val transformer: Llama2Transformer =
    if (useTensor)
      Llama2TensorTransformer.init(config, weights)
    else
      Llama2SimpleTransformer.init(config, weights)

  def run(): Unit = {
    val steps = 256

    var pos = 0
    var token = 1256
    var next = 0
    val start = System.nanoTime()
    while (pos < steps) {
      val logits = transformer.step(token, pos)
      next = logits.zipWithIndex.maxBy(_._1)._2 // argmax
      val tok = vocab.tokenScores(next)._1
      val tokenStr = if (token == 1 && tok == " ") tok.drop(1) else tok
      print(tokenStr)
      token = next
      pos += 1
    }
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