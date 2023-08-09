package net.virtualvoid.llama2

class Llama2Runner(transformer: Llama2Transformer, model: Llama2Model) {
  import model.vocab

  def iterate(steps: Int): Iterator[String] = new Iterator[String] {

    var pos = 0
    var token = 1

    def hasNext: Boolean = pos < steps && token != 0
    def next(): String = {
      val logits = transformer.step(token, pos)
      val next = logits.zipWithIndex.maxBy(_._1)._2 // argmax
      val tok = vocab.tokenScores(next)._1
      val tokenStr = if (token == 1 && tok == " ") tok.drop(1) else tok
      token = next
      pos += 1
      tokenStr
    }
  }
}
