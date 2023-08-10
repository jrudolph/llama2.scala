package net.virtualvoid.llama2

import scala.annotation.tailrec

class Llama2Runner(transformer: Llama2Transformer, model: Llama2Model) {
  import model.vocab

  def iterate(steps: Int, prompt: String = ""): Iterator[String] = new Iterator[String] {
    val promptTokens = bpeEncode(prompt)

    var pos = 0
    var token = 1

    def hasNext: Boolean = pos < steps && token != 0
    def next(): String = {
      val logits = transformer.step(token, pos)
      val next =
        if (pos < promptTokens.length) promptTokens(pos)
        else
          logits.zipWithIndex.maxBy(_._1)._2 // argmax
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
