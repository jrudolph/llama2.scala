package net.virtualvoid.llama2
package web

import java.io.File

case class ChooseToken(token: Int) extends Sampler {
  def sample(logits: Tensor1DMut): Int = token
}

trait Llama2State {
  def model: Llama2Model

  /** number of tokens processed up to this step */
  def sequenceLength: Int
  /** All preceding states up to this point including this */
  def stateHistory: Seq[Llama2State] // size of sequenceLength

  // Calculated k and v values for each layer in this step
  def k: Tensor2DMut // layer * dim
  def v: Tensor2DMut // layer * dim

  /** Output of the network at this step */
  def logits: Seq[Float]
  /** Output of sampling at this step */
  def chosenToken: Int

  /** sampler for this step */
  def sampler: Sampler

  def next(transformer: Llama2Transformer, sampler: Sampler): Llama2State

  def next(n: Int, transformer: Llama2Transformer, sampler: Sampler): Llama2State =
    if (n > 0) {
      //println(s"${state.sequenceLength} ${state.text}")
      val state = next(transformer, sampler)
      state.next(n - 1, transformer, sampler)
    } else this

  def prompt(nextTokens: Seq[Int]): Llama2State =
    if (nextTokens.isEmpty) this
    else {
      val n +: rem = nextTokens
      val nextState = next(Llama2TensorTransformer.init(model), ChooseToken(n))
      nextState.prompt(rem)
    }

  def chosenTokenText: String = model.vocab.tokenScores(chosenToken)._1
  def chosenTokenRank: Int = logits.zipWithIndex.sortBy(-_._1).indexWhere(_._2 == chosenToken)
  def text: String = stateHistory.map(i => model.vocab.tokenScores(i.chosenToken)._1).mkString
}

object Llama2State {
  def apply(_model: Llama2Model): Llama2State = new Llama2State {
    def model: Llama2Model = _model
    def sequenceLength: Int = 0
    def stateHistory: Seq[Llama2State] = Vector.empty
    def k: Tensor2DMut = ???
    def v: Tensor2DMut = ???
    def logits: Seq[Float] = ???
    val chosenToken: Int = 1
    val sampler: Sampler = ChooseToken(1)
    def next(transformer: Llama2Transformer, sampler: Sampler): Llama2State =
      Llama2State.next(_model, chosenToken, stateHistory, transformer, sampler)
  }

  def next(_model: Llama2Model, lastToken: Int, history: Seq[Llama2State], transformer: Llama2Transformer, _sampler: Sampler): Llama2State = {
    import _model.config._
    val pos = history.size
    val _k = Tensor2DMut.zero(nLayers, dim)
    val _v = Tensor2DMut.zero(nLayers, dim)

    val kv = new KV {
      def storeKey(layer: Int, t: Int, key: Tensor1DMut): Unit = {
        require(t == pos)
        _k(layer) := key
      }
      def storeValue(layer: Int, t: Int, key: Tensor1DMut): Unit = {
        require(t == pos)
        _v(layer) := key
      }

      def key(layer: Int, t: Int, idx: Int): Float =
        if (t == pos) _k.toFloatArray(layer * dim + idx)
        else history(t).k.toFloatArray(layer * dim + idx)

      def value(layer: Int, t: Int, idx: Int): Float =
        if (t == pos) _v.toFloatArray(layer * dim + idx)
        else history(t).v.toFloatArray(layer * dim + idx)
    }

    val _logits = transformer.step(lastToken, pos, kv)
    val _chosenToken = _sampler.sample(Tensor1DMut(_logits, _logits.size))

    new Llama2State {
      val model: Llama2Model = _model
      val sequenceLength: Int = pos + 1
      val stateHistory: Seq[Llama2State] = history :+ this
      val k: Tensor2DMut = _k
      val v: Tensor2DMut = _v
      val logits: Seq[Float] = _logits.toVector
      val chosenToken: Int = _chosenToken
      val sampler: Sampler = _sampler
      def next(transformer: Llama2Transformer, sampler: Sampler): Llama2State =
        Llama2State.next(model, chosenToken, stateHistory, transformer, sampler)
    }
  }
}

object Llama2StateTest /*extends App*/ {
  val baseDir = { // reStart runs with a subdirectory as the working directory
    val firstTry = new File("tokenizer.bin")
    if (firstTry.exists()) firstTry.getParentFile
    else new File("..")
  }
  //val checkpointFile = new File(baseDir, "llama2_7b.bin")
  val checkpointFile = new File(baseDir, "stories15M.bin")
  //val checkpointFile = new File(baseDir, "stories42M.bin")
  //val checkpointFile = new File(baseDir, "stories110M.bin")
  val tokenizerFile = new File(baseDir, "tokenizer.bin")

  val ggmlFile = new File(baseDir, "llama-2-7b.ggmlv3.q4_0.bin")

  val model = Llama2Model.fromLlama2CModel(checkpointFile, tokenizerFile)
  //val model = Llama2Model.fromGgml(ggmlFile)

  def printConfig(config: Config): Unit = {
    def p(name: String, value: Int): Unit = println(f"$name%-20s: $value%5d")

    p("Layers", config.nLayers)
    p("Dimension", config.dim)
    p("Hidden Dimension", config.hiddenDim)
    p("Heads", config.nHeads)
    p("Head Size", config.headSize)
    p("Vocab Size", config.vocabSize)
  }

  printConfig(model.config)

  val initial = Llama2State(model)
  val transformer = Llama2TensorTransformer.init(model)
  val sampler = TemperatureSampling(1f, 0.99f)

  def gen(state: Llama2State, remaining: Int): Llama2State =
    if (remaining > 0) {
      //println(s"${state.sequenceLength} ${state.text}")
      gen(state.next(transformer, sampler), remaining - 1)
    } else state

  val finalState = gen(initial, 200)
  println(finalState.stateHistory.map(h => s"${h.chosenTokenText} (${h.chosenTokenRank})").mkString)
}