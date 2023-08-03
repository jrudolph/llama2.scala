package net.virtualvoid.llama2

import java.io.{ File, FileInputStream, RandomAccessFile }
import java.nio.ByteOrder
import java.nio.channels.FileChannel.MapMode

object GgmlLoader {

  def fp16Tofp32(fp16: Short): Float = {
    // See https://github.com/Maratyszcza/FP16/blob/0a92994d729ff76a58f692d3028ca1b64b145d91/include/fp16/fp16.h#L108 / MIT license
    val w = (fp16.toInt & 0xffff) << 16
    val sign = w & 0x80000000
    val two_w = w + w

    val exp_offset = 0xE0 << 23
    val exp_scale = java.lang.Float.intBitsToFloat(0x7800000)

    val normalized_value = java.lang.Float.intBitsToFloat((two_w >> 4) + exp_offset) * exp_scale

    val magic_mask = 126 << 23
    val magic_bias = 0.5f
    val denormalized_value = java.lang.Float.intBitsToFloat((two_w >> 17) | magic_mask) - magic_bias

    val denormalized_cutoff = 1 << 27
    val result = sign |
      (if (two_w < denormalized_cutoff) java.lang.Float.floatToIntBits(denormalized_value) else java.lang.Float.floatToIntBits(normalized_value))
    java.lang.Float.intBitsToFloat(result)
  }

  def fromGgml(ggmlFile: File): (Config, Vocab, Weights) = {
    val f = new RandomAccessFile(ggmlFile, "r")

    def readIntLE(): Int =
      f.read() | (f.read() << 8) | (f.read() << 16) | (f.read() << 24)
    //val ch =
    val magic = readIntLE()
    val version = readIntLE()
    println(magic.toHexString)
    println(version)

    /*
    dim:           Int,
    hiddenDim:     Int,
    nLayers:       Int,
    nHeads:        Int,
    nKvHeads:      Int,
    vocabSize:     Int,
    seqLen:        Int,
    sharedWeights: Boolean,
    eps:           Float   = 1e-5f*/
    // 4096,11008,32,32,32,32000,2048,false,1.0E-5
    val vocabSize = readIntLE()
    val dim = readIntLE()
    val mult = readIntLE()
    val head = readIntLE()
    val layers = readIntLE()
    val rot = readIntLE()
    val ftype = readIntLE()

    val hiddenDim0 = (8 * dim) / 3
    val hiddenDim = ((hiddenDim0 + mult - 1) / mult) * mult

    println(s"vocabSize: $vocabSize dim: $dim mult: $mult head: $head layers: $layers rot: $rot ftype: $ftype hiddenDim: $hiddenDim")
    /* FIXME: for llama2-7b llama.cpp claims 512 but llama2.c 2048, this important mostly for the freq calculations (which might have been done wrong for llama2?) */
    val seqLen = 512
    val config = Config(dim, hiddenDim, layers, head, head, vocabSize, seqLen, false, 1e-5f)

    //val vocab = Vocab.fromFile(config, new File(ggmlFile.getParentFile, "tokenizer.bin"))
    def readVocab(): Vocab = {
      def readFloat(): Float = java.lang.Float.intBitsToFloat(readIntLE())

      val tokens =
        (0 until config.vocabSize).map { i =>
          val len = readIntLE()
          val bytes = new Array[Byte](len)
          f.read(bytes)
          val score = readFloat()
          (new String(bytes), score)
        }
      Vocab(tokens)
    }

    val vocab = readVocab()
    //vocab.tokenScores.drop(10000).take(100).foreach(println)

    val weightsBuffer = Map.newBuilder[String, AnyRef]

    while (f.getFilePointer < f.length) {
      val nDims = readIntLE()
      val nameLen = readIntLE()
      val tpe = readIntLE()
      val dims = (0 until nDims).map(_ => readIntLE())
      val nameBytes = new Array[Byte](nameLen)
      f.read(nameBytes)
      val name = new String(nameBytes)

      println(s"nDims: $nDims nameLen: $nameLen tpe: $tpe dims: $dims name: ${new String(nameBytes)}")

      println(s"Now at ${f.getFilePointer}")
      val padding = -(f.getFilePointer) & 31
      f.skipBytes(padding.toInt)
      println(s"Now at ${f.getFilePointer}")

      val size = tpe match {
        case 0 => dims.map(_.toLong).product * 4
        case 2 /* Q4_0 */ =>
          val qBlockSpace = 18L
          val qBlockK = 32L
          qBlockSpace * dims.map(_.toLong).product / qBlockK
      }
      println(s"size: $size")
      val dataOffset = f.getFilePointer

      val dataBuffer = f.getChannel.map(MapMode.READ_ONLY, dataOffset, size)
      val data = tpe match {
        case 0 =>
          val data = nDims match {
            case 1 => Tensor1D(dataBuffer.asFloatBuffer(), dims(0))
          }

          weightsBuffer += name -> data
        case 2 /* Q4_0 */ =>
          require(nDims == 2)
          val buffer = dataBuffer.duplicate()
          val shortBuffer = dataBuffer.order(ByteOrder.BIG_ENDIAN).asShortBuffer()
          val dim1 = dims(1)
          val dim2 = dims(0)

          val data =
            new Tensor2D {
              def size0: Int = dim1
              def size1: Int = dim2
              val K = 32

              def apply(i: Int): Tensor1D = new Tensor1D {
                def size: Int = dim2

                def copyToArray(dest: Array[Float], offset: Int): Unit = {
                  println(s"dest.length: ${dest.length} offset: $offset size: $size")
                  require(dest.length >= offset + size)
                  val numBlocksV = size / K

                  var j = 0
                  while (j < numBlocksV) {
                    val factor = quantizeFactor(i * size / K + j)

                    var k = 0
                    while (k < K / 2) {
                      val idx = i * size / 2 + j * K / 2 + k

                      val e = quantized(idx)

                      val x0 = (e & 0x0f) - 8
                      val x1 = ((e & 0xf0) >> 4) - 8

                      dest(offset + j * K + k) = x0 * factor
                      dest(offset + j * K + k + K / 2) = x1 * factor

                      //println(f"idx: $idx%5d e: $e%3d x0: $x0%3d x1: $x1%3d factor: $factor%1.5f ${dest(offset + j * K + k)}")

                      k += 1
                    }

                    //???
                    j += 1
                  }
                }

                def toFloatBuffer: FloatBuffer = ???
                def ∘(other: Tensor1DMut): Op1D = ???
              }

              def quantized(idx: Int): Byte = {
                // idx from 0 to size0 * size1 / 2
                val block = idx / K
                val blockOffset = idx % K
                val b = buffer.get(block * 18 + 2 /* d */ + blockOffset)
                //println(f"idx: $idx%5d block: $block%5d blockOffset: $blockOffset%5d b: $b%3d")
                b
              }

              def quantizeFactor(idx: Int): Float = {
                val bufIdx = idx * 18 / 2
                val fp16d = shortBuffer.get(bufIdx)
                val res = fp16Tofp32(fp16d)
                //println(f"idx: $idx%5d bufIdx: $bufIdx%5d fp16d: $fp16d%5d res: $res%1.5f")
                res
              }

              def `@`(v: Tensor1DMut): Op1D = { dest =>
                val arr = v.toFloatArray
                require(arr.length % K == 0)

                // quantize v as well (it will be used `dim1` times so it is worth it)
                val quantizedV = new Array[Byte](arr.size)
                val numBlocksV = arr.size / K
                val quantizeVFactor = new Array[Float](numBlocksV)

                {
                  var i = 0
                  while (i < numBlocksV) {
                    var j = 0
                    var max = 0f
                    while (j < K) {
                      val v = arr(i * K + j).abs
                      if (v > max) max = v
                      j += 1
                    }

                    val d = max / ((1 << 7) - 1)
                    val id = if (d != 0f) 1.0f / d else 0.0f

                    quantizeVFactor(i) = d
                    j = 0
                    while (j < K) {
                      val x0 = arr(i * K + j) * id // scale
                      quantizedV(i * K + j) = math.round(x0).toByte
                      j += 1
                    }

                    i += 1
                  }
                }

                var i = 0
                while (i < dim1) {
                  var j = 0
                  var sum = 0.0f
                  var sum2 = 0f
                  require(numBlocksV * 32 == dim2)
                  while (j < numBlocksV) {
                    val xF = quantizeFactor(i * dim2 / K + j)
                    val vF = quantizeVFactor(j)

                    var sumq = 0
                    var sumb = 0f
                    var k = 0
                    while (k < K / 2) {
                      /*const int v0 = (x[i].qs[j] & 0x0F) - 8;
                      const int v1 = (x[i].qs[j] >> 4) - 8;

                      sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk / 2]);*/
                      val idx = i * dim2 / 2 + j * K / 2 + k
                      val e = quantized(idx)
                      val x0 = (e & 0x0f) - 8
                      val x1 = ((e & 0xf0) >> 4) - 8

                      val v0 = quantizedV(j * K + k)
                      val v1 = quantizedV(j * K + k + K / 2)

                      /*if (i <= -1) {
                        val bufIdx = i * dim2 + j * K + k
                        println(f"idx: $idx%3d i: $i%3d j: $j%3d k: $k%3d bufIdx: $bufIdx%3d x0: $x0%3d x00: ${x0 * xF}%9.6f o0: ${floatBuffer.get(bufIdx)}%9.6f v0: $v0%3d v00: ${v0 * vF}%9.6f vo0: ${arr(j * K + k)}%9.6f")
                        println(f"ref: ${floatBuffer.get(i * dim2 + j * K + k) * arr(j * K + k)}%9.6f here: ${x0 * v0 * xF * vF}%9.6f")

                        println(f"idx: $idx%3d i: $i%3d j: $j%3d k: $k%3d bufIdx: $bufIdx%3d x1: $x1%3d x01: ${x1 * xF}%9.6f o0: ${floatBuffer.get(bufIdx + K / 2)}%9.6f v1: $v1%3d v01: ${v1 * vF}%9.6f vo1: ${arr(j * K + k + K / 2)}%9.6f")
                        println(f"ref: ${floatBuffer.get(i * dim2 + j * K + k + K / 2) * arr(j * K + k + K / 2)}%9.6f here: ${x1 * v1 * xF * vF}%9.6f")
                      }*/

                      //sumb += floatBuffer.get(i * dim2 + j * K + k) * arr(j * K + k) + floatBuffer.get(i * dim2 + j * K + k + K / 2) * arr(j * K + k + K / 2)
                      sumq += (x0 * v0) + (x1 * v1)
                      //sumq += quantized(i * dim2 + j * K + k) * quantizedV(j * K + k)
                      k += 1
                    }
                    //println(f"i: $i%3d j: $j%3d sumq: ${sumq.toFloat * xF * vF}%9.6f sumb: $sumb%9.6f")

                    sum += sumq.toFloat * xF * vF

                    if (i == -1)
                      ???

                    j += 1
                  }

                  dest(i) = sum
                  i += 1
                }
              }

              def toFloatArray: Array[Float] = ???
              def toFloatBuffer: FloatBuffer = ???
              def quantizeQ8: Tensor2D = ???
              def quantizeQ4: Tensor2D = ???
            }

          weightsBuffer += name -> data
      }

      f.seek(f.getFilePointer + size)
    }

    println("Found all weights")

    val allWeights = weightsBuffer.result()

    /*
    tok_embeddings.weight
    norm.weight
    output.weight
    layers.0.attention.wq.weight
    layers.0.attention.wk.weight
    layers.0.attention.wv.weight
    layers.0.attention.wo.weight
    layers.0.attention_norm.weight
    layers.0.feed_forward.w1.weight
    layers.0.feed_forward.w2.weight
    layers.0.feed_forward.w3.weight
    layers.0.ffn_norm.weight
    */

    def d1(name: String): Tensor1D = allWeights(name).asInstanceOf[Tensor1D]
    def d2(name: String): Tensor2D = allWeights(name).asInstanceOf[Tensor2D]
    def byLayerD1(name: String): Tensor2D = new Tensor2D {
      val layers = (0 until config.nLayers).map { i => d1(s"layers.$i.$name") }
      val dim = layers.head.size

      def size0: Int = config.nLayers
      def size1: Int = dim
      def apply(i: Int): Tensor1D = layers(i)
      def `@`(v: Tensor1DMut): Op1D = ???
      def toFloatArray: Array[Float] = ???
      def toFloatBuffer: FloatBuffer = ???
      def quantizeQ8: Tensor2D = ???
      def quantizeQ4: Tensor2D = ???
    }
    def byLayerD2(name: String): Tensor3D = new Tensor3D {
      val layers = (0 until config.nLayers).map { i => d2(s"layers.$i.$name") }
      val dim0 = layers.head.size0
      val dim1 = layers.head.size1

      override def size0: Int = config.nLayers
      override def size1: Int = dim0
      override def size2: Int = dim1

      override def apply(i: Int): Tensor2D = layers(i)

      def toFloatArray: Array[Float] = ???
      def toFloatBuffer: FloatBuffer = ???
    }

    // We load full model to extract the precomputed RoPE values
    // FIXME: recompute here
    val full7b = new File(ggmlFile.getParentFile, "llama2_7b.bin")
    val fullWeights = Weights.fromFile(Config.fromFile(full7b), full7b)

    val res =
      new Weights {
        val tokenEmbeddingTable: Tensor2D = d2("tok_embeddings.weight")
        val rms_att_weight: Tensor2D = byLayerD1("attention_norm.weight")
        val wq: Tensor3D = byLayerD2("attention.wq.weight")
        val wk: Tensor3D = byLayerD2("attention.wk.weight")
        val wv: Tensor3D = byLayerD2("attention.wv.weight")
        val wo: Tensor3D = byLayerD2("attention.wo.weight")
        val rms_ffn_weight: Tensor2D = byLayerD1("ffn_norm.weight")
        val w1: Tensor3D = byLayerD2("feed_forward.w1.weight")
        val w2: Tensor3D = byLayerD2("feed_forward.w2.weight")
        val w3: Tensor3D = byLayerD2("feed_forward.w3.weight")
        val rms_final_weight: Tensor1D = d1("norm.weight")
        val freq_cis_real: Tensor2D = fullWeights.freq_cis_real
        val freq_cis_imag: Tensor2D = fullWeights.freq_cis_imag
        val wcls: Tensor2D = d2("output.weight")
      }

    val t0 = Tensor1DMut.zero(config.vocabSize)
    val t1 = Tensor1DMut.zero(config.vocabSize)
    t0 := fullWeights.tokenEmbeddingTable(0)
    println(t0.toFloatArray.take(32).mkString(", "))
    t1 := res.tokenEmbeddingTable(0)
    println(t1.toFloatArray.take(32).mkString(", "))
    //???

    (config, vocab, res)
  }
}

object Llama2Main extends App {
  val baseDir = { // reStart runs with a subdirectory as the working directory
    val firstTry = new File("tokenizer.bin")
    if (firstTry.exists()) firstTry.getParentFile
    else new File("..")
  }
  //val checkpointFile = new File(baseDir, "llama2_7b.bin")
  //val checkpointFile = new File(baseDir, "stories15M.bin")
  //val checkpointFile = new File(baseDir, "stories42M.bin")
  val checkpointFile = new File(baseDir, "stories110M.bin")
  val tokenizerFile = new File(baseDir, "tokenizer.bin")

  val ggmlFile = new File(baseDir, "llama-2-7b.ggmlv3.q4_0.bin")

  val (config, vocab, weights) = GgmlLoader.fromGgml(ggmlFile)
  //val config = Config.fromFile(checkpointFile)
  //val vocab = Vocab.fromFile(config, tokenizerFile)
  //val weights = Weights.fromFile(config, checkpointFile)

  val useTensor = true
  val transformer: Llama2Transformer =
    if (useTensor)
      Llama2TensorTransformer.init(config, weights)
    else
      Llama2SimpleTransformer.init(config, weights)

  def run(): Unit = {
    val steps = 50

    var pos = 0
    var token = 1
    var next = 0
    val start = System.nanoTime()
    while (pos < steps) {
      val logits = transformer.step(token, pos)
      next = logits.zipWithIndex.maxBy(_._1)._2 // argmax
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
  run()
  run()
}