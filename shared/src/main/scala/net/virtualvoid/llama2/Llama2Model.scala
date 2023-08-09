package net.virtualvoid.llama2

import java.io.File

trait Llama2Model { outer =>
  val config: Config
  val vocab: Vocab
  val weights: Weights

  def quantizeQ4: Llama2Model = new Llama2Model {
    val config: Config = outer.config
    val vocab: Vocab = outer.vocab
    val weights: Weights = outer.weights.quantizeQ4
  }
  def quantizeQ8: Llama2Model = new Llama2Model {
    val config: Config = outer.config
    val vocab: Vocab = outer.vocab
    val weights: Weights = outer.weights.quantizeQ8
  }
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