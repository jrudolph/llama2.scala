package net.virtualvoid.llama2

import java.io.{ File, RandomAccessFile }
import java.nio.{ ByteOrder, FloatBuffer }
import java.nio.channels.FileChannel

trait Weights[nLayers <: Int, dim <: Int, hiddenDim <: Int] {
  def tokenEmbeddingTable: Tensor2D
  def rms_att_weight: Tensor2D
  def wq: Tensor3D[nLayers, dim, dim]
  def wk: Tensor3D[nLayers, dim, dim]
  def wv: Tensor3D[nLayers, dim, dim]
  def wo: Tensor3D[nLayers, dim, dim]
  def rms_ffn_weight: Tensor2D
  def w1: Tensor3D[nLayers, hiddenDim, dim]
  def w2: Tensor3D[nLayers, dim, hiddenDim]
  def w3: Tensor3D[nLayers, hiddenDim, dim]
  def rms_final_weight: Tensor1D
  def freq_cis_real: Tensor2D
  def freq_cis_imag: Tensor2D
  def wcls: Tensor2D
}
object Weights {
  def fromFile(config: Config, checkpointFile: File): Weights[config.nLayers.type, config.dim.type, config.hiddenDim.type] =
    Weights(config, Buffers.fromFile(checkpointFile, Config.HeaderSize))

  def apply(config: Config, buffers: Buffers): Weights[config.nLayers.type, config.dim.type, config.hiddenDim.type] = new Weights[config.nLayers.type, config.dim.type, config.hiddenDim.type] {
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
    val headSize = config.dim / config.nHeads
    val freq_cis_real = d2(config.seqLen, headSize / 2)
    val freq_cis_imag = d2(config.seqLen, headSize / 2)
    val wcls = if (config.sharedWeights) tokenEmbeddingTable else d2(config.vocabSize, config.dim)
  }
}