package net.virtualvoid.llama2

import java.io.File

/**
 * A one-to-one translation from llama2.c C code to primitive Scala code.
 *
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
class Llama2SimpleTransformer(
    config:         Config,
    weights:        Weights,
    val x:          Array[Float],
    val xb:         Array[Float],
    val xb2:        Array[Float],
    val hb:         Array[Float],
    val hb2:        Array[Float],
    val q:          Array[Float],
    val k:          Array[Float],
    val v:          Array[Float],
    val att:        Array[Float],
    val logits:     Array[Float],
    val keyCache:   Array[Float],
    val valueCache: Array[Float]
) extends Llama2Transformer {
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

  def step(token: Int, pos: Int, kv: KV, reporter: Reporter): Array[Float] = {
    import config._
    import weights._

    // copy embedding for token to x
    extractRowFrom2D(x, tokenEmbeddingTable, dim, token)

    // select freq rows for pos
    val freq_cis_real_row = freq_cis_real.duplicate().position(pos * headSize / 2).slice()
    val freq_cis_imag_row = freq_cis_imag.duplicate().position(pos * headSize / 2).slice()

    // for all layers
    var l = 0
    while (l < nLayers) {
      println(s"start layer $l")

      // attention rmsnorm
      rmsnorm(xb, x, select2d(rms_att_weight, dim, l), dim)

      trace1d("attention rmsnorm", xb)

      // qkv calculations
      matmul(q, xb, select3d(wq, dim, dim, l), dim, dim)
      matmul(k, xb, select3d(wk, dim, dim, l), dim, dim)
      matmul(v, xb, select3d(wv, dim, dim, l), dim, dim)

      trace1d("q", q)
      trace1d("k", k)
      trace1d("v", v)

      // apply RoPE rotation
      {
        var h = 0
        while (h < nHeads) {
          // get q and k for this head
          //val tq = q(h * headSize)
          //val tk = k(h * headSize)

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
      extractRow(keyCache, k, loff, pos, dim)
      extractRow(valueCache, v, loff, pos, dim)

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
                score += q(qOff + i) * keyCache(kCacheOff + i)
                i += 1
              }
              score /= math.sqrt(headSize).toFloat

              // safe to attention buffer
              att(attOffset + t) = score
              //println(f"att(${attOffset + t}%d) score $h%d $t%d $score%f")

              t += 1
            }
          }

          softmax(att, attOffset, pos + 1)
          trace1d("softmax att", att.take(pos + 1))

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
              val a = att(attOffset + t) // accumulate the weighted value into xb
              //println(f"a $h%d $t%d $a%f")

              //
              {
                var i = 0
                while (i < headSize) {
                  xb(xbOff + i) += valueCache(vOff + i) * a
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
      //println(s"after attention in layer $l")
      //xb.take(10).foreach(println)
      trace1d("attention", xb)

      // final matmul to get the output of the attention
      matmul(xb2, xb, select3d(wo, dim, dim, l), dim, dim)

      // residual connection
      accum(x, xb2, dim)

      trace1d("before ffn", x)

      // ffn rmsnorm
      rmsnorm(xb, x, select2d(rms_ffn_weight, dim, l), dim)

      // ffn
      matmul(hb, xb, select3d(w1, dim, hiddenDim, l), dim, hiddenDim)
      matmul(hb2, xb, select3d(w3, hiddenDim, dim, l), dim, hiddenDim)

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
      {
        var i = 0
        while (i < hiddenDim) {
          hb(i) *= hb2(i)
          i += 1
        }
      }

      // final matmul to get output of ffn
      matmul(xb, hb, select3d(w2, dim, hiddenDim, l), hiddenDim, dim)

      // residual connection
      accum(x, xb, dim)

      trace1d(s"layer done", x)

      l += 1
    }

    // final rmsnorm
    rmsnorm(x, x, rms_final_weight, dim)

    // classifier into logits
    matmul(logits, x, wcls, dim, vocabSize)

    logits
  }

  def extractRow(dest: Array[Float], src: Array[Float], loff: Int, pos: Int, dim: Int): Unit =
    System.arraycopy(src, 0, dest, loff + pos * dim, dim)

  def select3d(buffer: Tensor3D, dim1: Int, dim2: Int, i: Int): FloatBuffer = buffer(i)

  def select2d(buffer: FloatBuffer, dim1: Int, i: Int): FloatBuffer =
    buffer.duplicate().position(i * dim1).slice()

  def accum(into: Array[Float], from: Array[Float], size: Int): Unit = {
    var i = 0
    while (i < size) {
      into(i) += from(i)
      i += 1
    }
  }

  def softmax(x: Array[Float], off: Int, size: Int): Unit = {
    // find max value
    var max = x(off + 0)
    var j = 1
    while (j < size) {
      val v = x(off + j)
      if (v > max) max = v
      j += 1
    }

    // exp and sum
    var sum = 0.0f
    j = 0
    while (j < size) {
      val v = math.exp(x(off + j) - max).toFloat
      x(off + j) = v
      sum += v
      j += 1
    }

    // normalize
    j = 0
    while (j < size) {
      x(off + j) /= sum
      j += 1
    }
  }

  def rmsnorm(dest: Array[Float], x: Array[Float], weight: FloatBuffer, dim: Int): Unit = {
    // calculate sum of squares
    var sum = 0.0f
    var i = 0
    while (i < dim) {
      val v = x(i)
      sum += v * v
      i += 1
    }

    // calculate normalization factor
    val factor = 1f / math.sqrt(sum / dim + 1e-5f).toFloat

    // normalize and scale values
    i = 0
    while (i < dim) {
      dest(i) = x(i) * factor * weight.get(i)
      i += 1
    }
  }

  def matmul(dest: Array[Float], x: Array[Float], w: FloatBuffer, n: Int, d: Int): Unit = {
    var i = 0
    while (i < d) {
      var j = 0
      var sum = 0.0f
      while (j < n) {
        sum += w.get(i * n + j) * x(j)
        j += 1
      }
      dest(i) = sum
      i += 1
    }
  }

  def matmulTrace(dest: Array[Float], x: Array[Float], w: FloatBuffer, n: Int, d: Int): Unit = {
    var i = 0
    while (i < d) {
      var j = 0
      var sum = 0.0f
      while (j < n) {
        sum += w.get(i * n + j) * x(j)
        //if (i == 0) println(f"trace: $i, $j, ${w.get(i * n + j)}%f, ${x(j)}%f, $sum%f")
        j += 1
      }
      println(f"trace: $i: $sum")
      dest(i) = sum
      i += 1
    }
  }

  def extractRowFrom2D(dest: Array[Float], from: FloatBuffer, dim1: Int, at: Int): Unit = {
    val source = from.duplicate().position(at * dim1).slice()
    var i = 0
    while (i < dim1) {
      dest(i) = source.get(i)
      i += 1
    }
  }
}

object Llama2SimpleTransformer {
  def init(model: Llama2Model): Llama2Transformer = {
    import model._
    import config._
    new Llama2SimpleTransformer(
      config = config,
      weights = weights,
      x = new Array[Float](dim),
      xb = new Array[Float](dim),
      xb2 = new Array[Float](dim),
      hb = new Array[Float](hiddenDim),
      hb2 = new Array[Float](hiddenDim),
      q = new Array[Float](dim),
      k = new Array[Float](dim),
      v = new Array[Float](dim),
      att = new Array[Float](nHeads * seqLen),
      logits = new Array[Float](config.vocabSize),
      keyCache = new Array[Float](config.nLayers * seqLen * dim),
      valueCache = new Array[Float](config.nLayers * seqLen * dim)
    )
  }
}