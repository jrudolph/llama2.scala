package net.virtualvoid.llama2

import java.io.{ File, FileInputStream }

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
    dim:           Int,
    hiddenDim:     Int,
    nLayers:       Int,
    nHeads:        Int,
    nKvHeads:      Int,
    vocabSize:     Int,
    seqLen:        Int,
    sharedWeights: Boolean,
    eps:           Float   = 1e-5f
) {
  def headSize: Int = dim / nHeads
}
object Config {
  val HeaderSize = 7 * 4
  def fromFile(checkpoint: File): Config = {
    val fis = new FileInputStream(checkpoint)

    def readInt(): Int = {
      val b1 = fis.read()
      val b2 = fis.read()
      val b3 = fis.read()
      val b4 = fis.read()
      b1 | (b2 << 8) | (b3 << 16) | (b4 << 24)
    }

    val dim = readInt()
    val hiddenDim = readInt()
    val nLayers = readInt()
    val nHeads = readInt()
    val nKvHeads = readInt()
    val vocabSize = readInt()
    val seqLen = readInt()
    Config(dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize.abs, seqLen, vocabSize > 0)
  }
}