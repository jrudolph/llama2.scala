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
) {
  /*
  void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {

      // a few convenience variables
      float *x = s->x;
      int dim = p->dim;
      int hidden_dim =  p->hidden_dim;
      int head_size = dim / p->n_heads;

      // copy the token embedding into x
      float* content_row = &(w->token_embedding_table[token * dim]);
      memcpy(x, content_row, dim*sizeof(*x));

      // pluck out the "pos" row of freq_cis_real and freq_cis_imag
      float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
      float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

      // forward all the layers
      for(int l = 0; l < p->n_layers; l++) {

          // attention rmsnorm
          rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

          // qkv matmuls for this position
          matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
          matmul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
          matmul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

          // apply RoPE rotation to the q and k vectors for each head
          for (int h = 0; h < p->n_heads; h++) {
              // get the q and k vectors for this head
              float* q = s->q + h * head_size;
              float* k = s->k + h * head_size;
              // rotate q and k by the freq_cis_real and freq_cis_imag
              for (int i = 0; i < head_size; i+=2) {
                  float q0 = q[i];
                  float q1 = q[i+1];
                  float k0 = k[i];
                  float k1 = k[i+1];
                  float fcr = freq_cis_real_row[i/2];
                  float fci = freq_cis_imag_row[i/2];
                  q[i]   = q0 * fcr - q1 * fci;
                  q[i+1] = q0 * fci + q1 * fcr;
                  k[i]   = k0 * fcr - k1 * fci;
                  k[i+1] = k0 * fci + k1 * fcr;
              }
          }

          // save key,value at this time step (pos) to our kv cache
          int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
          float* key_cache_row = s->key_cache + loff + pos * dim;
          float* value_cache_row = s->value_cache + loff + pos * dim;
          memcpy(key_cache_row, s->k, dim*sizeof(*key_cache_row));
          memcpy(value_cache_row, s->v, dim*sizeof(*value_cache_row));

          // multihead attention. iterate over all heads
          int h;
          #pragma omp parallel for private(h)
          for (h = 0; h < p->n_heads; h++) {
              // get the query vector for this head
              float* q = s->q + h * head_size;
              // attention scores for this head
              float* att = s->att + h * p->seq_len;
              // iterate over all timesteps, including the current one
              for (int t = 0; t <= pos; t++) {
                  // get the key vector for this head and at this timestep
                  float* k = s->key_cache + loff + t * dim + h * head_size;
                  // calculate the attention score as the dot product of q and k
                  float score = 0.0f;
                  for (int i = 0; i < head_size; i++) {
                      score += q[i] * k[i];
                  }
                  score /= sqrtf(head_size);
                  // save the score to the attention buffer
                  att[t] = score;
              }

              // softmax the scores to get attention weights, from 0..pos inclusively
              softmax(att, pos + 1);

              // weighted sum of the values, store back into xb
              float* xb = s->xb + h * head_size;
              memset(xb, 0, head_size * sizeof(float));
              for (int t = 0; t <= pos; t++) {
                  // get the value vector for this head and at this timestep
                  float* v = s->value_cache + loff + t * dim + h * head_size;
                  // get the attention weight for this timestep
                  float a = att[t];
                  // accumulate the weighted value into xb
                  for (int i = 0; i < head_size; i++) {
                      xb[i] += a * v[i];
                  }
              }
          }

          // final matmul to get the output of the attention
          matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

          // residual connection back into x
          accum(x, s->xb2, dim);

          // ffn rmsnorm
          rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

          // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
          // first calculate self.w1(x) and self.w3(x)
          matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
          matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

          // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
          for (int i = 0; i < hidden_dim; i++) {
              s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
          }

          // elementwise multiply with w3(x)
          for (int i = 0; i < hidden_dim; i++) {
              s->hb[i] = s->hb[i] * s->hb2[i];
          }

          // final matmul to get the output of the ffn
          matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

          // residual connection
          accum(x, s->xb, dim);
      }

      // final rmsnorm
      rmsnorm(x, x, w->rms_final_weight, dim);

      // classifier into logits
      matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  }
   */
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
    matmul(logits, x, tokenEmbeddingTable, dim, vocabSize)
  }

  def extractRow(dest: Array[Float], src: Array[Float], loff: Int, pos: Int, dim: Int): Unit =
    System.arraycopy(src, 0, dest, loff + pos * dim, dim)

  def select3d(buffer: FloatBuffer, dim1: Int, dim2: Int, i: Int): FloatBuffer =
    buffer.duplicate().position(i * dim1 * dim2).slice()

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
    val source = from.duplicate().position(at * dim1)
    var i = 0
    while (i < dim1) {
      dest(i) = source.get()
      i += 1
    }
  }
}
object RunState {
  def init(config: Config): RunState = {
    import config._
    new RunState(
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

  val state = RunState.init(config)

  state.logits.take(100).foreach(println)

  var pos = 0
  var token = 1
  var next = 0
  while (pos < 100) {
    state.transformer(token, pos, config, weights)
    next = state.logits.zipWithIndex.maxBy(_._1)._2 // argmax
    val tok = vocab.tokenScores(next)._1
    val tokenStr = if (token == 1 && tok == " ") tok.drop(1) else tok
    print(tokenStr)
    token = next
    pos += 1
  }
}