package net.virtualvoid.llama2

import java.io.File

trait Llama2Model {
  val config: Config
  val vocab: Vocab
  val weights: Weights
}
object Llama2Model {
  def fromGgml(ggmlFile: File): Llama2Model =
    new Llama2Model {
      val (config, vocab, weights) = GgmlLoader.fromGgml(ggmlFile)
    }
  def fromLlama2CModel(checkpointFile: File, tokenizerFile: File): Llama2Model =
    new Llama2Model {
      val config = Config.fromFile(checkpointFile)
      val vocab = Vocab.fromFile(config, tokenizerFile)
      val weights = Weights.fromFile(config, checkpointFile)
    }
}