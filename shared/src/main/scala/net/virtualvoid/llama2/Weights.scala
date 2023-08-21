package net.virtualvoid.llama2

import java.io.{ File, RandomAccessFile }
import java.nio.{ ByteOrder, FloatBuffer }
import java.nio.channels.FileChannel

trait Weights {
  def tokenEmbeddingTable: Tensor2D
  def rms_att_weight: Tensor2D
  def wq: Tensor3D
  def wk: Tensor3D
  def wv: Tensor3D
  def wo: Tensor3D
  def rms_ffn_weight: Tensor2D
  def w1: Tensor3D
  def w2: Tensor3D
  def w3: Tensor3D
  def rms_final_weight: Tensor1D
  def freq_cis_real: Tensor2D
  def freq_cis_imag: Tensor2D
  def wcls: Tensor2D

  def quantizeQ4: Weights
  def quantizeQ8: Weights
}
object Weights {
  def fromFile(config: Config, checkpointFile: File): Weights =
    Weights(config, Buffers.fromFile(checkpointFile, Config.HeaderSize))

  def apply(config: Config, buffers: Buffers): Weights = new Weights { orig =>
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

    def quantizeQ4: Weights = new Weights {
      val tokenEmbeddingTable: Tensor2D = orig.tokenEmbeddingTable.quantizeQ4
      val rms_att_weight: Tensor2D = orig.rms_att_weight
      val wq: Tensor3D = orig.wq.quantizeQ4
      val wk: Tensor3D = orig.wk.quantizeQ4
      val wv: Tensor3D = orig.wv.quantizeQ4
      val wo: Tensor3D = orig.wo.quantizeQ4
      val rms_ffn_weight: Tensor2D = orig.rms_ffn_weight
      val w1: Tensor3D = orig.w1.quantizeQ4
      val w2: Tensor3D = orig.w2.quantizeQ4
      val w3: Tensor3D = orig.w3.quantizeQ4
      val rms_final_weight: Tensor1D = orig.rms_final_weight
      val freq_cis_real: Tensor2D = orig.freq_cis_real
      val freq_cis_imag: Tensor2D = orig.freq_cis_imag
      val wcls: Tensor2D = if (config.sharedWeights) tokenEmbeddingTable else orig.wcls.quantizeQ4

      def quantizeQ4: Weights = this
      def quantizeQ8: Weights = orig.quantizeQ8
    }
    def quantizeQ8: Weights = new Weights {
      val tokenEmbeddingTable: Tensor2D = orig.tokenEmbeddingTable.quantizeQ8
      val rms_att_weight: Tensor2D = orig.rms_att_weight
      val wq: Tensor3D = orig.wq.quantizeQ8
      val wk: Tensor3D = orig.wk.quantizeQ8
      val wv: Tensor3D = orig.wv.quantizeQ8
      val wo: Tensor3D = orig.wo.quantizeQ8
      val rms_ffn_weight: Tensor2D = orig.rms_ffn_weight
      val w1: Tensor3D = orig.w1.quantizeQ8
      val w2: Tensor3D = orig.w2.quantizeQ8
      val w3: Tensor3D = orig.w3.quantizeQ8
      val rms_final_weight: Tensor1D = orig.rms_final_weight
      val freq_cis_real: Tensor2D = orig.freq_cis_real
      val freq_cis_imag: Tensor2D = orig.freq_cis_imag
      val wcls: Tensor2D = if (config.sharedWeights) tokenEmbeddingTable else orig.wcls.quantizeQ8
      def quantizeQ4: Weights = orig.quantizeQ4
      def quantizeQ8: Weights = this
    }
  }
}