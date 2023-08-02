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
) {
  def headSize: Int = dim / nHeads
}

case class Vocab(
    tokenScores: Seq[(String, Float)]
)

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
}
object Weights {
  def apply(config: Config, buffer: FloatBuffer): Weights = new Weights {
    def d1(dim1: Int): Tensor1D = {
      val res = buffer.slice()
      res.limit(dim1)
      buffer.position(buffer.position() + dim1)
      Tensor1D(res, dim1)
    }
    def d2(dim1: Int, dim2: Int): Tensor2D = {
      val res = buffer.slice()
      res.limit(dim1 * dim2)
      buffer.position(buffer.position() + dim1 * dim2)
      Tensor2D(res, dim1, dim2)
    }
    def d3(dim1: Int, dim2: Int, dim3: Int): Tensor3D = {
      val res = buffer.slice()
      res.limit(dim1 * dim2 * dim3)
      buffer.position(buffer.position() + dim1 * dim2 * dim3)
      Tensor3D(res, dim1, dim2, dim3)
    }

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
  }
}

/**
 * @param x input
 * @param xb input with rmsnorm applied
 * @param xb2 input with rmsnorm applied twice
 * @param hb hidden state with rmsnorm applied
 * @param hb2 hidden state with rmsnorm applied twice
 * @param q query vector
 * @param k key vector
 * @param v value vector
 * @param att attention vector
 * @param logits logits vector
 * @param keyCache key cache for multiquery
 * @param valueCache value cache for multiquery
 */
class RunState(
    val x:          Tensor1D, // dim
    val xb:         Tensor1D, // dim
    val xb2:        Tensor1D, // dim
    val hb:         Tensor1D, // hiddenDim
    val hb2:        Tensor1D, // hiddenDim
    val q:          Tensor1D, // dim
    val k:          Tensor1D, // dim
    val v:          Tensor1D, // dim
    val att:        Tensor2D, // nHeads, seqLength
    val logits:     Tensor1D, // vocabSize
    val keyCache:   Tensor3D, // layer, seqLength, dim
    val valueCache: Tensor3D // layer, seqLength, dim
) {
  val writer = new java.io.PrintWriter(new File("output.txt"))
  def println(str: String): Unit = {
    //Console.println(str)
    writer.println(str)
    writer.flush()
  }
  def println(f: Float): Unit = println(f.toString)

  def trace1d(name: String, arr: Array[Float]): Unit = {
    //arr.zipWithIndex.foreach { case (x, i) => println(f"$name%s $i%5d $x%f") }
  }

  def transformer(token: Int, pos: Int, config: Config, weights: Weights): Unit = {
    import config._
    import weights._

    // copy embedding for token to x
    x := tokenEmbeddingTable(token)

    // select freq rows for pos
    val freq_cis_real_row = freq_cis_real(pos)
    val freq_cis_imag_row = freq_cis_imag(pos)

    // for all layers
    var l = 0
    while (l < nLayers) {
      println(s"start layer $l")

      // attention rmsnorm
      rmsnorm(xb, x, rms_att_weight(l))

      trace1d("attention rmsnorm", xb)

      // qkv calculations
      q := wq(l) `@` xb
      k := wk(l) `@` xb
      v := wv(l) `@` xb

      trace1d("q", q)
      trace1d("k", k)
      trace1d("v", v)

      // apply RoPE rotation
      {
        var h = 0
        while (h < nHeads) {
          // rotation
          {
            var i = 0
            while (i < headSize) {
              val q0 = q(h * headSize + i)
              val q1 = q(h * headSize + i + 1)
              val k0 = k(h * headSize + i)
              val k1 = k(h * headSize + i + 1)

              val fcr = freq_cis_real_row.get(i / 2)
              val fci = freq_cis_imag_row.get(i / 2)

              q(h * headSize + i) = q0 * fcr - q1 * fci
              q(h * headSize + i + 1) = q0 * fci + q1 * fcr
              k(h * headSize + i) = k0 * fcr - k1 * fci
              k(h * headSize + i + 1) = k0 * fci + k1 * fcr

              i += 2 // !
            }
          }

          h += 1
        }
      }

      trace1d("q_rope", q)
      trace1d("k_rope", k)
      trace1d("v_rope", v)

      // cache kv at this pos
      val loff = l * seqLen * dim
      keyCache(l)(pos) := k
      valueCache(l)(pos) := v

      // multihead attention
      {
        var h = 0
        while (h < nHeads) {
          val qOff = h * headSize
          val attOffset = h * seqLen

          // att(h)(t) = q(h) * keyCache(l)

          {
            var t = 0
            while (t <= /* ! */ pos) {
              val kCacheOff = loff + t * dim + h * headSize

              var score = 0f // q(h)
              var i = 0
              while (i < headSize) {
                score += q.toFloatArray(qOff + i) * keyCache.toFloatArray(kCacheOff + i)
                i += 1
              }
              score /= math.sqrt(headSize).toFloat

              // safe to attention buffer
              att(attOffset + t) = score
              //println(f"att(${attOffset + t}%d) score $h%d $t%d $score%f")

              t += 1
            }
          }

          softmax(att(h).shorten(pos + 1))
          trace1d("softmax att", att.toFloatArray.take(pos + 1))

          // weighted sum of the values, store into xb
          {
            val xbOff = h * headSize

            // reset xb row
            {
              var i = 0
              while (i < headSize) {
                xb(xbOff + i) = 0f
                i += 1
              }
            }
            // sum
            var t = 0
            while (t <= /* ! */ pos) {
              // get the value vector for this head and at this timestep
              val vOff = loff + t * dim + h * headSize
              // get the attention weight for this timestep
              val a = att.toFloatArray(attOffset + t) // accumulate the weighted value into xb
              //println(f"a $h%d $t%d $a%f")

              //
              {
                var i = 0
                while (i < headSize) {
                  xb(xbOff + i) += valueCache.toFloatArray(vOff + i) * a
                  //println(f"v $h%d $t%d $i%d ${valueCache(vOff + i)}%f $a%f ${xb(xbOff + i)}%f")
                  i += 1
                }
              }

              t += 1
            }
          }

          h += 1
        }
      }
      trace1d("attention", xb)

      // final matmul to get the output of the attention
      xb2 := wo(l) `@` xb

      // residual connection
      x += xb2

      trace1d("before ffn", x)

      // ffn rmsnorm
      rmsnorm(xb, x, rms_ffn_weight(l))

      // ffn
      hb := w1(l) `@` xb
      hb2 := w3(l) `@` xb

      // silu
      // hb := hb / (1f + exp(-hb))

      {
        var i = 0
        while (i < hiddenDim) {
          val x = hb(i)
          hb(i) = x / (1f + math.exp(-x)).toFloat
          i += 1
        }
      }

      // elementwise multiply
      hb ∘= hb2

      // final matmul to get output of ffn
      xb := w2(l) `@` hb

      // residual connection
      x += xb

      trace1d(s"layer done", x)

      l += 1
    }

    // final rmsnorm
    rmsnorm(x, x, rms_final_weight)

    // classifier into logits
    logits := tokenEmbeddingTable `@` x
  }

  def softmax(x: Tensor1D): Unit = {
    // find max value
    val max = x.max

    // exp and sum
    x -= max
    x.expMut()
    // normalize
    x /= x.sum
  }

  def rmsnorm(dest: Tensor1D, x: Tensor1D, weight: Tensor1D): Unit = {
    // calculate sum of squares
    val sum = x * x

    // scale values by weight
    dest := weight ∘ x

    // and normalize
    dest /= math.sqrt(sum / x.size + 1e-5f).toFloat
  }
}
object RunState {
  def init(config: Config): RunState = {
    import config._
    new RunState(
      x = Tensor1D.zero(dim),
      xb = Tensor1D.zero(dim),
      xb2 = Tensor1D.zero(dim),
      hb = Tensor1D.zero(hiddenDim),
      hb2 = Tensor1D.zero(hiddenDim),
      q = Tensor1D.zero(dim),
      k = Tensor1D.zero(dim),
      v = Tensor1D.zero(dim),
      att = Tensor2D.zero(nHeads, seqLen),
      logits = Tensor1D.zero(config.vocabSize),
      keyCache = Tensor3D.zero(config.nLayers, seqLen, dim),
      valueCache = Tensor3D.zero(config.nLayers, seqLen, dim)
    )
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
    val tokens =
      (0 until config.vocabSize).map { i =>
        val score = readFloat()
        val len = readInt()
        val bytes = new Array[Byte](len)
        fis.read(bytes)
        (new String(bytes), score)
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
  val vocab = readVocab(config, tokenizerFile)
  val weights = readWeights(config, checkpointFile)

  def run(): Unit = {
    val state = RunState.init(config)
    val steps = 256

    var pos = 0
    var token = 1
    var next = 0
    val start = System.nanoTime()
    while (pos < steps) {
      state.transformer(token, pos, config, weights)
      next = state.logits.toFloatArray.zipWithIndex.maxBy(_._1)._2 // argmax
      val tok = vocab.tokenScores(next)._1
      val tokenStr = if (token == 1 && tok == " ") tok.drop(1) else tok
      print(tokenStr)
      token = next
      pos += 1
    }
    println()

    val end = System.nanoTime()
    val lastedNanos = end - start
    val tokensPerSecond = steps.toFloat / lastedNanos * 1e9
    println(f"$tokensPerSecond%5.2f tokens per second")
  }
  run()
  run()
  run()
}