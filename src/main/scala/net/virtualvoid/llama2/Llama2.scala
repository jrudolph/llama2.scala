package net.virtualvoid.llama2

import java.io.{ File, FileInputStream, RandomAccessFile }
import java.nio.{ ByteOrder, FloatBuffer }
import java.nio.channels.FileChannel

/**
 * @param dim transformer dimension
 * @param hiddenDim for ffn layers
 * @param nLayers number of layers
 * @param nHeads number of query heads
 * @param nKvHeads number of key/value heads (can be < query heads because of multiquery)
 * @param vocabSize vocabulary size, usually 256 (byte-level)
 * @param seqLen max sequence length
 */
case class Config(
    dim:       Int,
    hiddenDim: Int,
    nLayers:   Int,
    nHeads:    Int,
    nKvHeads:  Int,
    vocabSize: Int,
    seqLen:    Int
)

case class Vocab(
    tokenScores: Seq[(String, Float)]
)

trait Weights {
  def tokenEmbeddingTable: FloatBuffer
  def rms_att_weight: FloatBuffer
  def wq: FloatBuffer
  def wk: FloatBuffer
  def wv: FloatBuffer
  def wo: FloatBuffer
  def rms_ffn_weight: FloatBuffer
  def w1: FloatBuffer
  def w2: FloatBuffer
  def w3: FloatBuffer
  def rms_final_weight: FloatBuffer
  def freq_cis_real: FloatBuffer
  def freq_cis_imag: FloatBuffer
}
object Weights {
  def apply(config: Config, buffer: FloatBuffer): Weights = new Weights {
    def d1(dim1: Int): FloatBuffer = {
      val res = buffer.slice()
      res.limit(dim1)
      buffer.position(buffer.position() + dim1)
      res
    }
    def d2(dim1: Int, dim2: Int): FloatBuffer = {
      val res = buffer.slice()
      res.limit(dim1 * dim2)
      buffer.position(buffer.position() + dim1 * dim2)
      res
    }
    def d3(dim1: Int, dim2: Int, dim3: Int): FloatBuffer = {
      val res = buffer.slice()
      res.limit(dim1 * dim2 * dim3)
      buffer.position(buffer.position() + dim1 * dim2 * dim3)
      res
    }

    val tokenEmbeddingTable: FloatBuffer = d2(config.vocabSize, config.dim)
    val rms_att_weight: FloatBuffer = d2(config.nLayers, config.dim)
    val wq: FloatBuffer = d3(config.nLayers, config.dim, config.dim)
    val wk: FloatBuffer = d3(config.nLayers, config.dim, config.dim)
    val wv: FloatBuffer = d3(config.nLayers, config.dim, config.dim)
    val wo: FloatBuffer = d3(config.nLayers, config.dim, config.dim)
    val rms_ffn_weight: FloatBuffer = d2(config.nLayers, config.dim)
    val w1: FloatBuffer = d3(config.nLayers, config.hiddenDim, config.dim)
    val w2: FloatBuffer = d3(config.nLayers, config.dim, config.hiddenDim)
    val w3: FloatBuffer = d3(config.nLayers, config.hiddenDim, config.dim)
    val rms_final_weight: FloatBuffer = d1(config.dim)
    val headSize = config.dim / config.nHeads
    val freq_cis_real: FloatBuffer = d2(config.seqLen, headSize / 2)
    val freq_cis_imag: FloatBuffer = d2(config.seqLen, headSize / 2)
  }
}

object Llama2Main extends App {
  val checkpointFile = new File("stories15M.bin")
  val tokenizerFile = new File("tokenizer.bin")

  val ConfigSize = 4 * 7

  def readConfig(checkpoint: File): Config = {
    val fis = new FileInputStream(checkpoint)

    def readInt(): Int = {
      val b1 = fis.read()
      val b2 = fis.read()
      val b3 = fis.read()
      val b4 = fis.read()
      b1 | (b2 << 8) | (b3 << 16) | (b4 << 24)
    }

    Config(
      dim = readInt(),
      hiddenDim = readInt(),
      nLayers = readInt(),
      nHeads = readInt(),
      nKvHeads = readInt(),
      vocabSize = readInt(),
      seqLen = readInt()
    )
  }

  def readVocab(config: Config, tokenizerFile: File): Vocab = {
    val fis = new FileInputStream(tokenizerFile)

    def readInt(): Int = {
      val b1 = fis.read()
      val b2 = fis.read()
      val b3 = fis.read()
      val b4 = fis.read()
      b1 | (b2 << 8) | (b3 << 16) | (b4 << 24)
    }

    def readFloat(): Float = java.lang.Float.intBitsToFloat(readInt())

    val maxTokenLength = readInt()
    println(s"Max token length: $maxTokenLength")
    val tokens =
      (0 until config.vocabSize).map { i =>
        val score = readFloat()
        val len = readInt()
        val bytes = new Array[Byte](len)
        fis.read(bytes)
        val res = (new String(bytes), score)
        //if (i < 1000) println(f"At $i%4d len $len%4d: $res")
        res
      }
    Vocab(tokens)
  }
  def readWeights(config: Config, checkpointFile: File): Weights = {
    // memory map the file and setup the weight buffers
    val raf = new RandomAccessFile(checkpointFile, "r")
    val buffer = raf.getChannel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    buffer.position(ConfigSize)
    val floatBuffer = buffer.asFloatBuffer()
    Weights(config, floatBuffer)
  }

  val config = readConfig(checkpointFile)
  println(config)
  val vocab = readVocab(config, tokenizerFile)
  vocab.tokenScores.take(10).foreach(println)
  val weights = readWeights(config, checkpointFile)
  println(f"token_embedding_table[0]: ${weights.tokenEmbeddingTable.get(0)}%g")
  println(f"rms_att_weight[0]: ${weights.rms_att_weight.get(0)}%g")
  println(f"wq[0]: ${weights.wq.get(0)}%g")
  println(f"wk[0]: ${weights.wk.get(0)}%g")
  println(f"wv[0]: ${weights.wv.get(0)}%g")
  println(f"wo[0]: ${weights.wo.get(0)}%g")
  println(f"rms_ffn_weight[0]: ${weights.rms_ffn_weight.get(0)}%g")
  println(f"w1[0]: ${weights.w1.get(0)}%g")
  println(f"w2[0]: ${weights.w2.get(0)}%g")
  println(f"w3[0]: ${weights.w3.get(0)}%g")
  println(f"rms_final_weight[0]: ${weights.rms_final_weight.get(0)}%g")
  println(f"freq_cis_real[0]: ${weights.freq_cis_real.get(0)}%g")
  println(f"freq_cis_imag[0]: ${weights.freq_cis_imag.get(0)}%g")
}