package net.virtualvoid.llama2

import java.io.File

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
class Llama2TensorTransformer(
    val config:  Config,
    val weights: Weights)(
    val x:          Tensor1DMut[weights.dim],
    val xb:         Tensor1DMut[weights.dim],
    val xb2:        Tensor1DMut[weights.dim],
    val hb:         Tensor1DMut[weights.hiddenDim],
    val hb2:        Tensor1DMut[weights.hiddenDim],
    val q:          Tensor1DMut[weights.dim],
    val k:          Tensor1DMut[weights.dim],
    val v:          Tensor1DMut[weights.dim],
    val att:        Tensor2DMut[weights.nHeads, weights.seqLen],
    val logits:     Tensor1DMut[weights.vocabSize],
    val keyCache:   Tensor3DMut[weights.nLayers, weights.seqLen, weights.dim],
    val valueCache: Tensor3DMut[weights.nLayers, weights.seqLen, weights.dim]
) extends Llama2Transformer {
  import config.eps

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

  def step(token: Int, pos: Int): Array[Float] = {
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
      xb := rmsnorm(x, rms_att_weight(l))

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
      xb := rmsnorm(x, rms_ffn_weight(l))

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
    x := rmsnorm(x, rms_final_weight)

    // classifier into logits
    logits := wcls `@` x

    logits
  }

  def softmax[d <: Int](x: Tensor1DMut[d]): Unit = {
    // find max value
    val max = x.max

    // exp and sum
    x -= max
    x.expMut()
    // normalize
    x /= x.sum
  }

  def rmsnorm[d <: Int](x: Tensor1DMut[d], weight: Tensor1D[d]): Op1D[d] = { dest =>
    // calculate sum of squares
    val sum = x * x

    // scale values by weight
    dest := weight ∘ x

    // and normalize
    dest /= math.sqrt(sum / x.size + eps).toFloat
  }
}
object Llama2TensorTransformer {
  def init(config: Config, weights: Weights): Llama2TensorTransformer = {
    import weights._
    new Llama2TensorTransformer(config, weights)(
      // casting?!? I couldn't figure out how else to do it, I think it has something to do with how opaque types work
      x = Tensor1DMut.zero(weights.dim).asInstanceOf[Tensor1DMut[weights.dim]],
      xb = Tensor1DMut.zero(dim).asInstanceOf[Tensor1DMut[weights.dim]],
      xb2 = Tensor1DMut.zero(dim).asInstanceOf[Tensor1DMut[weights.dim]],
      hb = Tensor1DMut.zero(hiddenDim).asInstanceOf[Tensor1DMut[weights.hiddenDim]],
      hb2 = Tensor1DMut.zero(hiddenDim).asInstanceOf[Tensor1DMut[weights.hiddenDim]],
      q = Tensor1DMut.zero(dim).asInstanceOf[Tensor1DMut[weights.dim]],
      k = Tensor1DMut.zero(dim).asInstanceOf[Tensor1DMut[weights.dim]],
      v = Tensor1DMut.zero(dim).asInstanceOf[Tensor1DMut[weights.dim]],
      att = Tensor2DMut.zero(nHeads, seqLen).asInstanceOf[Tensor2DMut[weights.nHeads, weights.seqLen]],
      logits = Tensor1DMut.zero(vocabSize).asInstanceOf[Tensor1DMut[weights.vocabSize]],
      keyCache = Tensor3DMut.zero(nLayers, seqLen, dim).asInstanceOf[Tensor3DMut[weights.nLayers, weights.seqLen, weights.dim]],
      valueCache = Tensor3DMut.zero(nLayers, seqLen, dim).asInstanceOf[Tensor3DMut[weights.nLayers, weights.seqLen, weights.dim]]
    )
  }
}