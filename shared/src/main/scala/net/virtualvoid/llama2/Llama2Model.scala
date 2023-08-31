package net.virtualvoid.llama2

import java.io.File

trait Llama2Model { outer =>
  def name: String
  val config: Config
  val vocab: Vocab
  val weights: Weights

  def quantizeQ4: Llama2Model = new Llama2Model {
    val name = outer.name + "_q4"
    val config: Config = outer.config
    val vocab: Vocab = outer.vocab
    val weights: Weights = outer.weights.quantizeQ4
  }
  def quantizeQ8: Llama2Model = new Llama2Model {
    val name = outer.name + "_q8"
    val config: Config = outer.config
    val vocab: Vocab = outer.vocab
    val weights: Weights = outer.weights.quantizeQ8
  }
}
object Llama2Model {
  def fromGgml(ggmlFile: File): Llama2Model =
    new Llama2Model {
      val name = ggmlFile.getName.reverse.dropWhile(_ != '.').drop(1).reverse
      val (config, vocab, weights) = GgmlLoader.fromGgml(ggmlFile)
    }
  def fromLlama2CModel(checkpointFile: File, tokenizerFile: File): Llama2Model =
    new Llama2Model {
      val name = checkpointFile.getName.reverse.dropWhile(_ != '.').drop(1).reverse
      val config = Config.fromFile(checkpointFile)
      val vocab = Vocab.fromFile(config, tokenizerFile)
      val weights = Weights.fromFile(config, checkpointFile)
    }
}