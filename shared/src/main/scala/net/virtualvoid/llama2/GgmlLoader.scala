package net.virtualvoid.llama2

import java.io.{ File, RandomAccessFile }
import java.nio.ByteOrder
import java.nio.channels.FileChannel.MapMode

object GgmlLoader {

  def fromGgml(ggmlFile: File): (Config, Vocab, Weights) = {
    val f = new RandomAccessFile(ggmlFile, "r")

    def readIntLE(): Int =
      f.read() | (f.read() << 8) | (f.read() << 16) | (f.read() << 24)

    val magic = readIntLE()
    val version = readIntLE()

    require(magic == 0x67676a74, "Only ggml files with magic 0x67676a74 supported")
    require(version == 3, "Only ggml version 3 supported")

    val vocabSize = readIntLE()
    val dim = readIntLE()
    val mult = readIntLE()
    val head = readIntLE()
    val layers = readIntLE()
    val rot = readIntLE()
    val ftype = readIntLE()

    val hiddenDim0 = (8 * dim) / 3
    val hiddenDim = ((hiddenDim0 + mult - 1) / mult) * mult

    //println(s"vocabSize: $vocabSize dim: $dim mult: $mult head: $head layers: $layers rot: $rot ftype: $ftype hiddenDim: $hiddenDim")
    /* FIXME: for llama2-7b llama.cpp claims 512 but llama2.c 2048, this important mostly for the freq calculations (which might have been done wrong for llama2?) */
    val seqLen = 2048
    val config = Config(dim, hiddenDim, layers, head, head, vocabSize, seqLen, false, 1e-5f)

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

    val weightsBuffer = Map.newBuilder[String, AnyRef]

    while (f.getFilePointer < f.length) {
      val startOfBlock = f.getFilePointer
      val nDims = readIntLE()
      val nameLen = readIntLE()
      val tpe = readIntLE()
      val dims = (0 until nDims).map(_ => readIntLE())
      val nameBytes = new Array[Byte](nameLen)
      f.read(nameBytes)
      val name = new String(nameBytes)

      //println(f"name: $name%-40s startOfBlock: $startOfBlock nDims: $nDims nameLen: $nameLen tpe: $tpe dims: $dims")

      //println(s"Now at ${f.getFilePointer}")
      val padding = -(f.getFilePointer) & 31
      f.skipBytes(padding.toInt)
      //println(s"Start of data for $name at ${f.getFilePointer}")

      val size = tpe match {
        case 0 => dims.map(_.toLong).product * 4
        case 2 /* Q4_0 */ =>
          val qBlockSpace = 18L
          val qBlockK = 32L
          qBlockSpace * dims.map(_.toLong).product / qBlockK

        case n => throw new IllegalArgumentException(s"Unsupported type for parameters '$name': $n")
      }

      val dataOffset = f.getFilePointer

      val dataBuffer = f.getChannel.map(MapMode.READ_ONLY, dataOffset, size)
      tpe match {
        case 0 =>
          val data = nDims match {
            case 1 => Tensor1D(FloatBuffer.fromNioBuffer(dataBuffer.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()), dims(0))
            case n => throw new IllegalArgumentException(s"Unsupported number of dimensions for FP32 parameters '$name': $n")
          }

          weightsBuffer += name -> data
        case 2 /* Q4_0 */ =>
          require(nDims == 2, s"Unsupported number of dimensions for Q4_0 parameters '$name': $nDims")
          val buffer = dataBuffer
          val shortBuffer = dataBuffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
          val dim1 = dims(1)
          val dim2 = dims(0)

          val data =
            new Tensor2D {
              def size0: Int = dim1
              def size1: Int = dim2
              val K = ScalaMathImplementation.QK4_0

              def apply(i: Int): Tensor1D = new Tensor1D {
                def size: Int = dim2

                def copyToArray(dest: Array[Float], offset: Int): Unit = {
                  require(dest.length >= offset + size)
                  val numBlocksV = size / K
                  val blockOffset = i * size / K

                  var j = 0
                  while (j < numBlocksV) {
                    val blockId = blockOffset + j
                    val factor = quantizeFactor(blockId)

                    var k = 0
                    while (k < K / 2) {
                      val e = quantized(blockId, k)

                      val x0 = (e & 0x0f) - 8
                      val x1 = ((e & 0xf0) >> 4) - 8

                      dest(offset + j * K + k) = x0 * factor
                      dest(offset + j * K + k + K / 2) = x1 * factor

                      k += 1
                    }

                    j += 1
                  }
                }

                def toFloatBuffer: FloatBuffer = ???
                def ∘(other: Tensor1DMut): Op1D = ???
              }

              def quantized(block: Int, offset: Int): Byte =
                buffer.get(block * 18 + 2 /* d */ + offset)

              def quantizeFactor(idx: Int): Float = {
                val bufIdx = idx * 18 / 2
                val fp16d = shortBuffer.get(bufIdx)
                ScalaMathImplementation.fp16Tofp32(fp16d)
              }

              val quantizedV = new Array[Byte](dim2)
              val numBlocksV = dim2 / K
              val quantizeVFactor = new Array[Float](numBlocksV)

              def `@`(v: Tensor1DMut): Op1D = { dest =>
                val arr = v.toFloatArray
                require(arr.length % K == 0)

                // quantize v as well (it will be used `dim1` times so it is worth it)

                //val start = System.nanoTime()

                MathImplementation.Default.quantizeQ8(arr, quantizedV, quantizeVFactor)
                MathImplementation.Default.matMulQ4Q8FromBuffer(buffer, quantizedV, quantizeVFactor, dim1, dim2, dest)

              }

              def toFloatArray: Array[Float] = ???
              def toFloatBuffer: FloatBuffer = ???
              def quantizeQ8: Tensor2D = ???
              def quantizeQ4: Tensor2D = this
            }

          weightsBuffer += name -> data
      }

      f.seek(f.getFilePointer + size)
    }

    val allWeights = weightsBuffer.result()

    def d1(name: String): Tensor1D = allWeights(name).asInstanceOf[Tensor1D]
    def d2(name: String): Tensor2D = allWeights(name).asInstanceOf[Tensor2D]
    def byLayerD1(name: String): Tensor2D = new Tensor2D {
      val layers = (0 until config.nLayers).map { i => d1(s"layers.$i.$name") }.toArray
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
      val layers = (0 until config.nLayers).map { i => d2(s"layers.$i.$name") }.toArray
      val dim0 = layers.head.size0
      val dim1 = layers.head.size1

      def size0: Int = config.nLayers
      def size1: Int = dim0
      def size2: Int = dim1

      def apply(i: Int): Tensor2D = layers(i)

      def quantizeQ4: Tensor3D = throw new IllegalArgumentException("Model can not be requantized to Q4_0")
      def quantizeQ8: Tensor3D = throw new IllegalArgumentException("Model can not be requantized to Q8_0")

      def toFloatArray: Array[Float] = ???
      def toFloatBuffer: FloatBuffer = ???
    }

    // We load full model to extract the precomputed RoPE values
    // FIXME: recompute here
    val full7b = new File(ggmlFile.getParentFile, "llama2_7b.bin")
    val fullConfig = Config.fromFile(full7b)
    val fullWeights = Weights.fromFile(fullConfig, full7b)

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

        def quantizeQ4: Weights = throw new IllegalArgumentException("Not supported")
        def quantizeQ8: Weights = throw new IllegalArgumentException("Not supported")
      }

    (config, vocab, res)
  }
}