package net.virtualvoid.llama2

import java.io.{ File, RandomAccessFile }
import java.nio.{ ByteOrder, FloatBuffer }
import java.nio.channels.FileChannel

trait Weights {
  type dim <: Int
  type hiddenDim <: Int
  type nLayers <: Int
  type nHeads <: Int
  type vocabSize <: Int
  type seqLen <: Int
  type headSize <: Int
  type halfHeadSize <: Int
  val dim: dim
  val hiddenDim: hiddenDim
  val nLayers: nLayers
  val nHeads: nHeads
  val vocabSize: vocabSize
  val seqLen: seqLen
  val headSize: headSize
  val halfHeadSize: halfHeadSize

  def tokenEmbeddingTable: Tensor2D[vocabSize, dim]
  def rms_att_weight: Tensor2D[nLayers, dim]
  def wq: Tensor3D[nLayers, dim, dim]
  def wk: Tensor3D[nLayers, dim, dim]
  def wv: Tensor3D[nLayers, dim, dim]
  def wo: Tensor3D[nLayers, dim, dim]
  def rms_ffn_weight: Tensor2D[nLayers, dim]
  def w1: Tensor3D[nLayers, hiddenDim, dim]
  def w2: Tensor3D[nLayers, dim, hiddenDim]
  def w3: Tensor3D[nLayers, hiddenDim, dim]
  def rms_final_weight: Tensor1D[dim]
  def freq_cis_real: Tensor2D[seqLen, halfHeadSize]
  def freq_cis_imag: Tensor2D[seqLen, halfHeadSize]
  def wcls: Tensor2D[vocabSize, dim]
}
object Weights {
  def fromFile(config: Config, checkpointFile: File): Weights =
    Weights(config, Buffers.fromFile(checkpointFile, Config.HeaderSize))

  def apply(config: Config, buffers: Buffers): Weights = new Weights {
    type dim = config.dim.type
    type hiddenDim = config.hiddenDim.type
    type nLayers = config.nLayers.type
    type nHeads = config.nHeads.type
    type vocabSize = config.vocabSize.type
    type seqLen = config.seqLen.type
    type headSize = config.headSize.type
    type halfHeadSize = config.halfHeadSize.type

    val dim: config.dim.type = config.dim
    val hiddenDim: config.hiddenDim.type = config.hiddenDim
    val nLayers: config.nLayers.type = config.nLayers
    val nHeads: config.nHeads.type = config.nHeads
    val vocabSize: config.vocabSize.type = config.vocabSize
    val seqLen: config.seqLen.type = config.seqLen
    val headSize: config.headSize.type = config.headSize
    val halfHeadSize: config.halfHeadSize.type = config.halfHeadSize

    import buffers._

    val tokenEmbeddingTable = d2(config.vocabSize, config.dim)
    val rms_att_weight = d2(config.nLayers, config.dim)
    val wq = d3(config.nLayers, config.dim, config.dim)
    val wk = d3(config.nLayers, config.dim, config.dim)
    val wv = d3(config.nLayers, config.dim, config.dim)
    val wo = d3(config.nLayers, config.dim, config.dim)
    val rms_ffn_weight = d2(config.nLayers, config.dim)
    val w1 = d3(config.nLayers, config.hiddenDim, config.dim)
    val w2 = d3(config.nLayers, config.dim, config.hiddenDim)
    val w3 = d3(config.nLayers, config.hiddenDim, config.dim)
    val rms_final_weight = d1(config.dim)
    val freq_cis_real = d2(config.seqLen, config.halfHeadSize)
    val freq_cis_imag = d2(config.seqLen, config.halfHeadSize)
    val wcls = if (config.sharedWeights) tokenEmbeddingTable else d2(config.vocabSize, config.dim)
  }
}