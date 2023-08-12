package net.virtualvoid.llama2

import scala.annotation.tailrec
import scala.util.Random

trait Sampler {
  def sample(logits: Tensor1DMut): Int
}
object ArgmaxSampler extends Sampler {
  def sample(logits: Tensor1DMut): Int = logits.toFloatArray.zipWithIndex.maxBy(_._1)._2
}

class TemperatureSampling(temperature: Float, topp: Float = 1f, random: Random = new Random, vocab: Vocab) extends Sampler {
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

      if (topp != 1f) {
        def selectBySorting(): Int = {
          val sorted = logits.toFloatArray.zipWithIndex.sortBy(-_._1).toVector
          val cumSum = sorted.iterator.scanLeft(0f)(_ + _._1)
          val idx = cumSum.indexWhere(_ > topp)
          val interesting = Tensor1DMut(sorted.take(idx).map(_._1).toArray, idx)
          interesting /= interesting.sum
          sorted(sample(interesting))._2
        }

        /**
         * This method tries to avoid full on sorting of the big logits array.
         *
         * The idea is to exploit properties of the distribution array:
         *  1. we expect that the top-k values are much larger than the rest (i.e. an inverse power-law distribution)
         *  2. values need to add up to 1
         *
         * 1. means that the top-p entries are few, so scanning is reasonable compared to sorting
         * 2. each scan can be aborted early if we found a value that accounts for more than half of the remaining probability
         */
        def selectByScanning(): Int = {
          // FIXME: needs work if there are multiple values of prevMax
          val vs = logits.toFloatArray
          // values smaller than (1 - topp) / (logits.length - 1) cannot be part of the result, so sort them out directly
          val cutoff = (1f - topp) / (logits.length - 1)
          val idxs = vs.zipWithIndex.filter(_._1 >= cutoff).map(_._2)

          def maxLessThan(remaining: Float, prevMax: Float): Int = {
            val halfRemaining = remaining / 2
            var maxIndex = idxs(0)
            var max = vs(maxIndex)
            var i = 0
            while (i < idxs.length && max < halfRemaining) {
              val idx = idxs(i)
              val v = vs(idx)
              if (v > max && v < prevMax) {
                max = v
                maxIndex = idx
              }
              i += 1
            }
            maxIndex
          }

          val scanAttempts = 100
          val idxBuffer = new Array[Int](scanAttempts)
          val pBuffer = new Array[Float](scanAttempts)

          def collect(numFound: Int, sum: Float, prevMax: Float): Int =
            if (sum > topp) numFound
            else if (numFound >= idxBuffer.size) -1
            else {
              val maxIdx = maxLessThan(1f - sum, prevMax)
              val v = vs(maxIdx)
              idxBuffer(numFound) = maxIdx
              pBuffer(numFound) = v
              collect(numFound + 1, sum + v, v)
            }

          val numFound = collect(0, 0f, 2f)
          if (numFound == -1) selectBySorting()
          else {
            val interesting = Tensor1DMut(pBuffer.take(numFound), numFound)
            interesting /= interesting.sum
            val sid = sample(interesting)
            idxBuffer(sid)
          }
        }
        selectByScanning()
      } else
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

      /* interesting to see what possible completions are
      if (pos == -1) {
        val myLogits = Tensor1DMut.zero(logits.size)
        myLogits := Tensor1DMut(logits, logits.size)
        myLogits.softmaxMut()
        println()
        myLogits.toFloatArray.zipWithIndex.sortBy(-_._1).take(10).foreach {
          case (p, t) =>
            println(f"${vocab.tokenScores(t)._1}%-20s ${p * 100}%5.2f %%")
        }
      }*/

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
