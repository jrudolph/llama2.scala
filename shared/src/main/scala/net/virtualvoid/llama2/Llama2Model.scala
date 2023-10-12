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
  def fromComponents(_config: Config, _vocab: Vocab, _weights: Weights): Llama2Model =
    new Llama2Model {
      val (config, vocab, weights) = (_config, _vocab, _weights)
    }
  def fromComponents(x: (Config, Vocab, Weights)): Llama2Model =
    fromComponents(x._1, x._2, x._3)

  def fromGgml(ggmlFile: File): Llama2Model =
    fromComponents(GgmlLoader.fromGgml(ggmlFile))

  def fromGguf(ggufFile: File): Llama2Model =
    fromComponents(GgufLoader.fromGguf(ggufFile))

  def fromLlama2CModel(checkpointFile: File, tokenizerFile: File): Llama2Model = {
    val config = Config.fromFile(checkpointFile)
    fromComponents(
      config,
      Vocab.fromFile(config, tokenizerFile),
      Weights.fromFile(config, checkpointFile)
    )
  }
}