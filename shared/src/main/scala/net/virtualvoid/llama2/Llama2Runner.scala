package net.virtualvoid.llama2

import scala.annotation.tailrec
import scala.util.Random

trait Sampler {
  def sample(logits: Tensor1DMut): Int
}
object ArgmaxSampler extends Sampler {
  def sample(logits: Tensor1DMut): Int = logits.toFloatArray.zipWithIndex.maxBy(_._1)._2
}

class TemperatureSampling(temperature: Float, random: Random = new Random) extends Sampler {
  override def sample(logits: Tensor1DMut): Int = {

    def sample(tensor1DMut: Tensor1DMut): Int = {
      val p = random.nextFloat()
      tensor1DMut.toFloatArray
        .iterator
        .zipWithIndex
        .scanLeft((0f, 0)) { case (sum, i) => (sum._1 + i._1, i._2) }
        .dropWhile(_._1 < p)
        .next()._2
    }

    if (temperature == 0f)
      logits.toFloatArray.zipWithIndex.maxBy(_._1)._2 // argmax
    else {
      logits /= temperature
      logits.softmaxMut()
      sample(logits)
    }
  }
}

class Llama2Runner(transformer: Llama2Transformer, model: Llama2Model) {
  import model.vocab

  def iterate(steps: Int, sampler: Sampler = ArgmaxSampler, prompt: String = ""): Iterator[String] = new Iterator[String] {
    val promptTokens = bpeEncode(prompt)

    var pos = 0
    var token = 1

    def hasNext: Boolean = pos < steps && token != 0
    def next(): String = {
      val logits = transformer.step(token, pos)
      val next =
        if (pos < promptTokens.length) promptTokens(pos)
        else
          sampler.sample(Tensor1DMut(logits, logits.size))
      val tok = vocab.tokenScores(next)._1
      val tokenStr = if (token == 1 && tok == " ") tok.drop(1) else tok
      token = next
      pos += 1
      tokenStr
    }
  }

  def bpeEncode(prompt: String): Seq[Int] = {
    val tokMap = vocab.tokenScores.zipWithIndex.map { case ((t, s), i) => t -> (i, s) }.toMap
    var toks: Vector[Int] = prompt.map(x => tokMap(x.toString)._1).toVector

    @tailrec def mergeTokensStep(toks: Vector[Int]): Vector[Int] = {
      val candidates =
        toks.sliding(2).zipWithIndex.flatMap {
          case (Seq(t1: Int, t2: Int), i: Int) =>
            val tok = vocab.tokenScores(t1)._1 + vocab.tokenScores(t2)._1
            val id = tokMap.get(tok)
            id.map { case (newTok, score) => (i, newTok, score) }
        }

      if (candidates.isEmpty) toks
      else {
        val (idx, newTok, _) = candidates.maxBy(_._3)
        mergeTokensStep(toks.take(idx) ++ Seq(newTok) ++ toks.drop(idx + 2))
      }
    }

    mergeTokensStep(toks)
  }
}
