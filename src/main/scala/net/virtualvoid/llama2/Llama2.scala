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
    dim:           Int,
    hiddenDim:     Int,
    nLayers:       Int,
    nHeads:        Int,
    nKvHeads:      Int,
    vocabSize:     Int,
    seqLen:        Int,
    sharedWeights: Boolean
) {
  def headSize: Int = dim / nHeads
}

case class Vocab(
    tokenScores: Seq[(String, Float)]
)

trait Tensor1D {
  def size: Int

  def max: Float
  def sum: Float
  def *(other: Tensor1D): Float
  def +=(other: Tensor1D): Unit
  def +=(scalar: Float): Unit
  def -=(scalar: Float): Unit = this += -scalar

  def *=(scalar: Float): Unit
  def /=(scalar: Float): Unit = this *= 1f / scalar

  /* element-wise multiplication */
  def *=(other: Tensor1D): Unit

  def expMut(): Unit

  def elementWiseMulInto(other: Tensor1D, dest: Tensor1D): Unit

  /** Returns a new tensor backed by the same data but assuming a different dimension */
  def shorten(newDim: Int): Tensor1D
  def copyTo(dest: Tensor1D): Unit

  def toFloatArray: Array[Float]
  def toFloatBuffer: FloatBuffer
}
object Tensor1D {
  def zero(dim1: Int): Tensor1D = Tensor1D(new Array[Float](dim1), dim1)
  def apply(floatBuffer: FloatBuffer, dim1: Int): Tensor1D = new Tensor1D {
    def size: Int = dim1

    def max: Float = ???
    def sum: Float = ???

    def *(other: Tensor1D): Float = ???
    def +=(other: Tensor1D): Unit = ???
    def +=(scalar: Float): Unit = ???
    def *=(scalar: Float): Unit = ???
    def *=(other: Tensor1D): Unit = ???

    def expMut(): Unit = ???

    def elementWiseMulInto(other: Tensor1D, dest: Tensor1D): Unit = {
      require(other.size == this.size)
      val destArray = dest.toFloatArray
      val others = other.toFloatArray
      var i = 0
      while (i < size) {
        destArray(i) = floatBuffer.get(i) * others(i)
        i += 1
      }
    }

    def shorten(newDim: Int): Tensor1D = ???

    def copyTo(dest: Tensor1D): Unit = {
      // FIXME: assuming that dest has toFloatArray
      floatBuffer.duplicate().get(dest.toFloatArray, 0, dim1)
    }
    def toFloatArray: Array[Float] = ???
    def toFloatBuffer: FloatBuffer = floatBuffer
  }

  def apply(fs: Array[Float], dim: Int, offset: Int = 0): Tensor1D = new Tensor1D {
    require(fs.size >= offset + dim)

    def floats(i: Int): Float = fs(offset + i)

    def size: Int = dim

    def max: Float = {
      var max = floats(0)
      var i = 1
      while (i < dim) {
        val v = floats(i)
        if (v > max) max = v
        i += 1
      }
      max
    }
    def sum: Float = {
      var sum = 0f
      var i = 0
      while (i < dim) {
        sum += floats(i)
        i += 1
      }
      sum
    }

    def *(other: Tensor1D): Float = {
      require(other.size == this.size)
      val others = other.toFloatArray
      var i = 0
      var sum = 0f
      while (i < size) {
        sum += floats(i) * others(i)
        i += 1
      }
      sum
    }
    def +=(other: Tensor1D): Unit = {
      val others = other.toFloatArray
      var i = 0
      while (i < dim) {
        fs(offset + i) += others(i)
        i += 1
      }
    }

    def +=(scalar: Float): Unit = {
      var i = 0
      while (i < dim) {
        fs(offset + i) += scalar
        i += 1
      }
    }

    def *=(scalar: Float): Unit = {
      var i = 0
      while (i < dim) {
        fs(offset + i) *= scalar
        i += 1
      }
    }

    def *=(other: Tensor1D): Unit = {
      val others = other.toFloatArray
      var i = 0
      while (i < dim) {
        fs(offset + i) *= others(i)
        i += 1
      }
    }

    def expMut(): Unit = {
      var i = 0
      while (i < dim) {
        fs(offset + i) = math.exp(floats(i)).toFloat
        i += 1
      }
    }

    def elementWiseMulInto(other: Tensor1D, dest: Tensor1D): Unit = ???

    def shorten(newDim: Int): Tensor1D = {
      require(newDim <= size)
      Tensor1D(fs, newDim, offset)
    }

    def copyTo(dest: Tensor1D): Unit = ???

    def toFloatArray: Array[Float] =
      if (offset == 0 && fs.size == dim) fs
      else ???

    def toFloatBuffer: FloatBuffer = ???
  }

  implicit def autoBuffer(t1: Tensor1D): FloatBuffer = t1.toFloatBuffer
  implicit def autoArray(t1: Tensor1D): Array[Float] = t1.toFloatArray
}
trait Tensor2D {
  def size0: Int
  def size1: Int

  def apply(i: Int): Tensor1D

  def mulInto(v: Tensor1D, dest: Tensor1D): Tensor1D

  def toFloatArray: Array[Float]
  def toFloatBuffer: FloatBuffer
}
object Tensor2D {
  def zero(dim1: Int, dim2: Int): Tensor2D = Tensor2D(new Array[Float](dim1 * dim2), dim1, dim2)
  def apply(fb: FloatBuffer, dim1: Int, dim2: Int): Tensor2D = new Tensor2D {
    val floatBuffer = fb.duplicate()
    def size0: Int = dim1
    def size1: Int = dim2
    def apply(i: Int): Tensor1D = {
      val source = floatBuffer.duplicate().position(i * dim2).slice()
      Tensor1D(source, dim2)
    }
    def mulInto(v: Tensor1D, dest: Tensor1D): Tensor1D = {
      require(v.size == dim2)
      var i = 0
      while (i < dim1) {
        var j = 0
        var sum = 0.0f
        while (j < dim2) {
          sum += floatBuffer.get(i * dim2 + j) * v(j)
          j += 1
        }
        dest(i) = sum
        i += 1
      }
      dest
    }

    def toFloatArray: Array[Float] = ???
    def toFloatBuffer: FloatBuffer = floatBuffer.duplicate()
  }

  def apply(floats: Array[Float], dim1: Int, dim2: Int): Tensor2D = new Tensor2D {
    require(floats.size == dim1 * dim2)
    def size0: Int = dim1
    def size1: Int = dim2

    def apply(i: Int): Tensor1D = Tensor1D(floats, dim2, offset = i * dim2)
    override def mulInto(v: Tensor1D, dest: Tensor1D): Tensor1D = ???

    def toFloatArray: Array[Float] = floats
    def toFloatBuffer: FloatBuffer = ???
  }

  implicit def autoBuffer(t2: Tensor2D): FloatBuffer = t2.toFloatBuffer
  implicit def autoArray(t2: Tensor2D): Array[Float] = t2.toFloatArray
}
trait Tensor3D {
  def size0: Int

  def size1: Int
  def size2: Int

  def apply(i: Int): Tensor2D

  def toFloatArray: Array[Float]
  def toFloatBuffer: FloatBuffer
}
object Tensor3D {
  def zero(dim1: Int, dim2: Int, dim3: Int): Tensor3D = Tensor3D(new Array[Float](dim1 * dim2 * dim3), dim1, dim2, dim3)
  def apply(floatBuffer: FloatBuffer, dim1: Int, dim2: Int, dim3: Int): Tensor3D = new Tensor3D {
    def size0: Int = dim1
    def size1: Int = dim2
    def size2: Int = dim3

    def apply(i: Int): Tensor2D = {
      val source = floatBuffer.duplicate().position(i * dim2 * dim3).slice()
      Tensor2D(source, dim2, dim3)
    }

    def toFloatArray: Array[Float] = ???
    def toFloatBuffer: FloatBuffer = floatBuffer.duplicate()
  }

  def apply(floats: Array[Float], dim1: Int, dim2: Int, dim3: Int): Tensor3D = new Tensor3D {
    require(floats.size == dim1 * dim2 * dim3)

    def size0: Int = dim1
    def size1: Int = dim2
    def size2: Int = dim3

    def apply(i: Int): Tensor2D = ???

    def toFloatArray: Array[Float] = floats
    def toFloatBuffer: FloatBuffer = ???
  }
  implicit def autoBuffer(t3: Tensor3D): FloatBuffer = t3.toFloatBuffer
  implicit def autoArray(t3: Tensor3D): Array[Float] = t3.toFloatArray
}

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
  def apply(config: Config, ch: FileChannel): Weights = new Weights {
    var pos = 7L * 4L // skip config header

    def buffer(numElements: Int): FloatBuffer = {
      val buffer = ch.map(FileChannel.MapMode.READ_ONLY, pos, numElements * 4)
      pos += numElements * 4
      buffer.order(ByteOrder.LITTLE_ENDIAN)
      buffer.asFloatBuffer
    }

    def d1(dim1: Int): Tensor1D = Tensor1D(buffer(dim1), dim1)
    def d2(dim1: Int, dim2: Int): Tensor2D = Tensor2D(buffer(dim1 * dim2), dim1, dim2)
    def d3(dim1: Int, dim2: Int, dim3: Int): Tensor3D = Tensor3D(buffer(dim1 * dim2 * dim3), dim1, dim2, dim3)

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
    tokenEmbeddingTable(token).copyTo(x)

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
      wq(l).mulInto(xb, q)
      wk(l).mulInto(xb, k)
      wv(l).mulInto(xb, v)

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
      storeRow(keyCache, k, loff, pos, dim)
      storeRow(valueCache, v, loff, pos, dim)

      // multihead attention
      {
        var h = 0
        while (h < nHeads) {
          val qOff = h * headSize
          val attOffset = h * seqLen

          {
            var t = 0
            while (t <= /* ! */ pos) {
              val kCacheOff = loff + t * dim + h * headSize

              var score = 0f
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
      wo(l).mulInto(xb, xb2)

      // residual connection
      x += xb2

      trace1d("before ffn", x)

      // ffn rmsnorm
      rmsnorm(xb, x, rms_ffn_weight(l))

      // ffn
      w1(l).mulInto(xb, hb)
      w3(l).mulInto(xb, hb2)

      // silu

      {
        var i = 0
        while (i < hiddenDim) {
          val x = hb(i)
          hb(i) = x / (1f + math.exp(-x)).toFloat
          i += 1
        }
      }

      // elementwise multiply
      hb *= hb2

      // final matmul to get output of ffn
      w2(l).mulInto(hb, xb)

      // residual connection
      x += xb

      trace1d(s"layer done", x)

      l += 1
    }

    // final rmsnorm
    rmsnorm(x, x, rms_final_weight)

    // classifier into logits
    tokenEmbeddingTable.mulInto(x, logits)
  }

  def storeRow(dest: Array[Float], src: Array[Float], loff: Int, pos: Int, dim: Int): Unit =
    System.arraycopy(src, 0, dest, loff + pos * dim, dim)

  def softmax(x: Tensor1D): Unit = {
    // find max value
    val max = x.max

    // exp and sum
    x -= max
    x.expMut()
    val sum = x.sum
    // normalize
    x /= sum
  }

  def rmsnorm(dest: Tensor1D, x: Tensor1D, weight: Tensor1D): Unit = {
    // calculate sum of squares
    val sum = x * x

    // scale values by weight
    weight.elementWiseMulInto(x, dest)
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

    val dim = readInt()
    val hiddenDim = readInt()
    val nLayers = readInt()
    val nHeads = readInt()
    val nKvHeads = readInt()
    val vocabSize = readInt()
    val seqLen = readInt()
    Config(dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize.abs, seqLen, vocabSize > 0)
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
    val raf = new RandomAccessFile(checkpointFile, "r")
    Weights(config, raf.getChannel)
  }

  val config = readConfig(checkpointFile)
  println(config)
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