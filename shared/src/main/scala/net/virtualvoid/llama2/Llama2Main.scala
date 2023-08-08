package net.virtualvoid.llama2

import java.io.File
import scala.util.Random

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
    val temp = .5f
    val seed = new Random().nextLong()
    val random = new Random(seed)
    val steps = 256

    var pos = 0
    var token = 1
    var next = 0
    val start = System.nanoTime()
    while (pos < steps) {
      val logits = transformer.step(token, pos)

      def softmax(x: Tensor1DMut): Unit = {
        // find max value
        val max = x.max

        // exp and sum
        x -= max
        x.expMut()
        // normalize
        x /= x.sum
      }
      def sample(tensor1DMut: Tensor1DMut): Int = {
        val p = random.nextFloat()
        tensor1DMut.toFloatArray
          .iterator
          .zipWithIndex
          .scanLeft((0f, 0)) { case (sum, i) => (sum._1 + i._1, i._2) }
          .dropWhile(_._1 < p)
          .next()._2
      }

      next =
        if (temp == 0f)
          logits.zipWithIndex.maxBy(_._1)._2 // argmax
        else {
          val ls = Tensor1DMut(logits, logits.size)
          ls /= temp
          softmax(ls)
          sample(ls)
        }

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
  run()
  run()
  run()
  run()
  run()
}