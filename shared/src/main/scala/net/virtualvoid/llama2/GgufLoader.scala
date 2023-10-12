package net.virtualvoid.llama2

import java.io.{File, RandomAccessFile}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel.MapMode
import scala.reflect.ClassTag

/**
 * Gguf reader based on the preliminary spec at
 * https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
 */
object GgufLoader {

  def fromGguf(ggufFile: File): (Config, Vocab, Weights) = {
    val f = new RandomAccessFile(ggufFile, "r")

    def readIntLE(): Int =
      f.read() | (f.read() << 8) | (f.read() << 16) | (f.read() << 24)
    def readLongLE(): Long =
      (readIntLE() & 0xffffffffL) | ((readIntLE() & 0xffffffffL) << 32)
    def readString(): String =
      readLongLE().toInt match {
        case 0 => ""
        case len =>
          val bytes = new Array[Byte](len)
          f.read(bytes)
          new String(bytes, "utf8")
      }
    def readFloat32LE(): Float =
      java.lang.Float.intBitsToFloat(readIntLE())

    val magic = readIntLE()
    val version = readIntLE()

    require(magic == 0x46554747, "Only gguf files with magic 0x67676a74 supported")
    require(version == 2, "Version must be 2")

    val tensorCount = readLongLE()
    val metadataEntries = readLongLE()

    enum GgmlType(val tag: Int) {
      case F32 extends GgmlType(0)
      case F16 extends GgmlType(1)
      case Q4_0 extends GgmlType(2)
      case Q4_1 extends GgmlType(3)
      case Q5_0 extends GgmlType(6)
      case Q5_1 extends GgmlType(7)
      case Q8_0 extends GgmlType(8)
      case Q8_1 extends GgmlType(9)
      case Q2_K extends GgmlType(10)
      case Q3_K extends GgmlType(11)
      case Q4_K extends GgmlType(12)
      case Q5_K extends GgmlType(13)
      case Q6_K extends GgmlType(14)
      case Q8_K extends GgmlType(15)
      case I8 extends GgmlType(16)
      case I16 extends GgmlType(17)
      case I32 extends GgmlType(18)
    }
    object GgmlType {
      def byTag = values.map(v => v.tag -> v).toMap
    }

    enum MetadataValueType(val tag: Int) {
      case UInt8 extends MetadataValueType(0)
      case Int8 extends MetadataValueType(1)
      case UInt16 extends MetadataValueType(2)
      case Int16 extends MetadataValueType(3)
      case UInt32 extends MetadataValueType(4)
      case Int32 extends MetadataValueType(5)
      case Float32 extends MetadataValueType(6)
      case Bool extends MetadataValueType(7)
      case String extends MetadataValueType(8)
      case Array extends MetadataValueType(9)
      case UInt64 extends MetadataValueType(10)
      case Int64 extends MetadataValueType(11)
      case Float64 extends MetadataValueType(12)
    }
    object MetadataValueType {
      def byTag = values.map(v => v.tag -> v).toMap
    }

    enum MetadataValue {
      case Integer(value: Long) extends MetadataValue
      case Str(value: String) extends MetadataValue
      case Float32(value: Float) extends MetadataValue
      case Float64(value: Double) extends MetadataValue
      case Bool(value: Boolean) extends MetadataValue
      case Array(tpe: MetadataValueType, values: Seq[MetadataValue]) extends MetadataValue

      def cast[T <: MetadataValue: ClassTag]: T = this match {
        case t: T => t
        case _ => throw new IllegalArgumentException(s"Can not convert $this to ${ClassTag[T]}")
      }

      def longValue: Long = cast[Integer].value
      def intValue: Int = cast[Integer].value.toInt // FIXME: overflow check
      def stringValue: String = cast[Str].value
      def floatValue: Float = cast[Float32].value
      def booleanValue: Boolean = cast[Bool].value
    }
    def readValue(tpe: MetadataValueType): MetadataValue =
      tpe match {
        case MetadataValueType.String => MetadataValue.Str(readString())
        case MetadataValueType.UInt32 => MetadataValue.Integer(readIntLE() & 0xffffffffffffffffL)
        case MetadataValueType.Int32 => MetadataValue.Integer(readIntLE())
        case MetadataValueType.Float32 => MetadataValue.Float32(readFloat32LE())
        case MetadataValueType.Array =>
          val tpe = MetadataValueType.byTag(readIntLE())
          val len = readLongLE().toInt
          val values = Seq.fill(len)(readValue(tpe))
          MetadataValue.Array(tpe, values)
      }

    def readKV(): (String, MetadataValue) = {
      val key = readString()
      val tpe = MetadataValueType.byTag(readIntLE())
      val value = readValue(tpe)

      if (tpe != MetadataValueType.Array)
        println(s"key: $key tpe: $tpe value: $value")
      else {
        val arr = value.asInstanceOf[MetadataValue.Array]
        println(s"key: $key tpe: $tpe value: Array(${arr.tpe}, ${arr.values.size})")
      }
      key -> value
    }

    case class TensorInfo(
      name: String,
      numDimensions: Int,
      dimensions: Seq[Long],
      dataType: GgmlType,
      offset: Long
    )
    def readTensorInfo(): TensorInfo = {
      val name = readString()
      val numDims = readIntLE()
      val dims = Seq.fill(numDims)(readLongLE())
      val tpe = GgmlType.byTag(readIntLE())
      val offset = readLongLE()

      val info = TensorInfo(name, numDims, dims, tpe, offset)
      println(info)
      info
    }

    //println(f"tensorCount: $tensorCount metadataEntries: $metadataEntries")

    val metadata = Seq.fill(metadataEntries.toInt)(readKV()).toMap
    val tensorInfos = Seq.fill(tensorCount.toInt)(readTensorInfo()).map(info => info.name -> info).toMap

    val architecture = metadata("general.architecture")
    require(architecture.stringValue == "llama", s"Only llama architecture supported but got '$architecture'")

    def metadataInt(name: String): Int = metadata(name).intValue

    val dim = metadataInt("llama.embedding_length")
    val hiddenDim = metadataInt("llama.feed_forward_length")
    val layers = metadataInt("llama.block_count")
    val head = metadataInt("llama.attention.head_count")
    //val headKV = metadataInt("llama.attention.head_count_kv")
    val eps = metadata("llama.attention.layer_norm_rms_epsilon").floatValue
    val ropeDimensionCount = metadataInt("llama.rope.dimension_count")
    val ropeFreqBase = metadata.get("llama.rope.freq_base").map(_.floatValue).getOrElse(10000)
    val seqLen = metadataInt("llama.context_length") min 2048
    val vocabSize = metadata("tokenizer.ggml.tokens").cast[MetadataValue.Array].values.size

    val config = Config(dim, hiddenDim, layers, head, head, vocabSize, seqLen, false, eps)

    def readVocab(): Vocab = {
      val tokenizerType = metadata("tokenizer.ggml.model").stringValue
      require(tokenizerType == "llama", s"Only support llama style tokenizer for now but got '$tokenizerType'")

      val toks = metadata("tokenizer.ggml.tokens").cast[MetadataValue.Array].values.map(_.stringValue).toSeq.map(_.replace('▁', ' '))
      val scores = metadata("tokenizer.ggml.scores").cast[MetadataValue.Array].values.map(_.floatValue).toSeq

      Vocab(toks.zip(scores))
    }

    val vocab = readVocab()

    val alignment = metadata.get("general.alignment").map(_.intValue).getOrElse(32)
    val quantizationVersion = metadataInt("general.quantization_version")
    require(quantizationVersion == 2, s"Only quantization version 2 supported but got $quantizationVersion")

    val start = f.getFilePointer
    // align with alignment (which might not be a power of 2), so use modulo, must be a positive value
    val padding = (alignment - (start % alignment)) % alignment
    val offset = start + padding

    def sizeOfType(tpe: GgmlType, dims: Long): Long = tpe match {
      case GgmlType.F32 => dims * 4
      case GgmlType.Q4_0 =>
        val qBlockSpace = 18L
        val qBlockK = 32L
        qBlockSpace * dims / qBlockK
      case GgmlType.Q6_K =>
        val qBlockSpace = 210L
        val qBlockK = 256L
        val size = qBlockSpace * dims / qBlockK
        // println(f"Q6_K size: $size dims: $dims qBlockSpace: $qBlockSpace qBlockK: $qBlockK")
        size
    }


    def d1(name: String): Tensor1D = {
      val info = tensorInfos(name)
      require(info.numDimensions == 1, s"1D tensor expected for $name, but got ${info.numDimensions}")
      require(info.dataType == GgmlType.F32, s"Only F32 tensors supported, but got ${info.dataType} for $name")

      val elements = info.dimensions.product
      val size = sizeOfType(info.dataType, elements)
      val dataBuffer = f.getChannel.map(MapMode.READ_ONLY, offset + info.offset, size)

      Tensor1D(FloatBuffer.fromNioBuffer(dataBuffer.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()), info.dimensions(0).toInt)
    }

    def tensor2dQ4_0(buffer: ByteBuffer, info: TensorInfo): Tensor2D = {
      val shortBuffer = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
      val dims = info.dimensions.map(_.toInt)
      val dim1 = dims(1)
      val dim2 = dims(0)

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
    }
    def tensor2DQ6_K(dataBuffer: ByteBuffer, info: TensorInfo): Tensor2D = {
      def dequantizeQ6_K(dataBuffer: ByteBuffer, info: TensorInfo): FloatBuffer = {
        val numEls = info.dimensions.product.toInt
        val size = numEls * 4
        val buffer = ByteBuffer.allocateDirect(size).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

        val QK_K = 256
        require(numEls % QK_K == 0)
        val numBlocks = numEls / QK_K
        // structure of a block_q6_K
        val blockSize = 210
        val qlOff = 0
        val qhOff = qlOff + 128
        val scalesOff = qhOff + 64
        val dOff = scalesOff + 16

        var blockIdx = 0
        var y = 0
        while(blockIdx < numBlocks) {
          val offset = blockSize * blockIdx
          def u8(i: Int): Int = dataBuffer.get(offset + i) & 0xff
          def i8(i: Int): Int = dataBuffer.get(offset + i).toInt

          val d = ScalaMathImplementation.fp16Tofp32(
            (u8(dOff) | (u8(dOff + 1) << 8)).toShort
          )

          var n = 0
          while (n < 2) {
            var l = 0
            while (l < 32) {
              val is = l / 16
              val qloff = qlOff + n * 64 + l
              val qhl = u8(qhOff + n * 32 + l)
              val soff = scalesOff + n * 8 + is
              val q1 = ((u8(qloff + 0) & 0xf) | (((qhl >> 0) & 3) << 4)) - 32
              val q2 = ((u8(qloff + 32) & 0xf) | (((qhl >> 2) & 3) << 4)) - 32
              val q3 = ((u8(qloff + 0) >> 4) | (((qhl >> 4) & 3) << 4)) - 32
              val q4 = ((u8(qloff + 32) >> 4) | (((qhl >> 6) & 3) << 4)) - 32

              buffer.put(y + l + 0, d * i8(soff + 0) * q1)
              buffer.put(y + l + 32, d * i8(soff + 2) * q2)
              buffer.put(y + l + 64, d * i8(soff + 4) * q3)
              buffer.put(y + l + 96, d * i8(soff + 6) * q4)

              l += 1
            }
            y += 128

            n += 1
          }

          blockIdx+= 1
        }

        buffer
      }

      val data = dequantizeQ6_K(dataBuffer, info)
      Tensor2D(data, info.dimensions(1).toInt, info.dimensions(0).toInt)
    }

    def d2(name: String): Tensor2D = {
      val info = tensorInfos(name)
      require(info.numDimensions == 2, s"2D tensor expected for $name, but got ${info.numDimensions}")

      val elements = info.dimensions.product
      val size = sizeOfType(info.dataType, elements)
      val dataBuffer = f.getChannel.map(MapMode.READ_ONLY, offset + info.offset, size)

      info.dataType match {
        case GgmlType.Q4_0 => tensor2dQ4_0(dataBuffer, info)
        case GgmlType.Q6_K => tensor2DQ6_K(dataBuffer, info)
      }
    }

    def byLayerD1(name: String): Tensor2D = new Tensor2D {
      val layers = (0 until config.nLayers).map { i => d1(s"blk.$i.$name") }.toArray
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
      val layers = (0 until config.nLayers).map { i => d2(s"blk.$i.$name") }.toArray
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
    val full7b = new File(ggufFile.getParentFile, "llama2_7b.bin")
    val fullConfig = Config.fromFile(full7b)
    val fullWeights = Weights.fromFile(fullConfig, full7b)

    // FIXME: something is slightly off with the Q6_K weights
    // maybe it's not? Since we haven't implemented Q6_K_Q8_K matmul but just dequantize, we will have different kinds
    // of rounding errors than the original model when executed on llama.cpp

    // you can use this to compare with other quantizations
    /*
    val ggmlModel = Llama2Model.fromGgml(new File("../llama-2-7b.ggmlv3.q4_0.bin"))
    val gw = ggmlModel.weights

    def dequantize(t: Tensor2D): Tensor2D = {
      val data = new Array[Float](t.size0 * t.size1)
      (0 until t.size0).foreach { i =>
        t(i).copyToArray(data, i * t.size1)
      }
      val fb = ByteBuffer.allocateDirect(4 * t.size0 * t.size1).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
      fb.put(data)
      fb.flip()
      Tensor2D(fb, t.size0, t.size1)
    }

    println(gw.wcls.size0 -> gw.wcls.size1)
    val wcn = dequantize(d2("output.weight"))
    val buf1 = new Array[Float](wcn.size0 * wcn.size1)
    wcn.toFloatBuffer.get(buf1, 0, wcn.size0 * wcn.size1)

    println(wcn.size0 -> wcn.size1)
    val ggmlwcn = fullWeights.wcls//dequantize(ggmlModel.weights.wcls)
    val buf2 = new Array[Float](ggmlwcn.size0 * ggmlwcn.size1)
    ggmlwcn.toFloatBuffer.get(buf2, 0, ggmlwcn.size0 * ggmlwcn.size1)

    buf1.zip(buf2).foreach { case (a, b) =>
      println(a -> b)
    }

     */

    val weights =
      new Weights {
          val tokenEmbeddingTable: Tensor2D = d2("token_embd.weight")
          val rms_att_weight: Tensor2D = byLayerD1("attn_norm.weight")
          val wq: Tensor3D = byLayerD2("attn_q.weight")
          val wk: Tensor3D = byLayerD2("attn_k.weight")
          val wv: Tensor3D = byLayerD2("attn_v.weight")
          val wo: Tensor3D = byLayerD2("attn_output.weight")
          val rms_ffn_weight: Tensor2D = byLayerD1("ffn_norm.weight")
          val w1: Tensor3D = byLayerD2("ffn_gate.weight")
          val w2: Tensor3D = byLayerD2("ffn_down.weight")
          val w3: Tensor3D = byLayerD2("ffn_up.weight")
          val rms_final_weight: Tensor1D = d1("output_norm.weight")
          val freq_cis_real: Tensor2D = fullWeights.freq_cis_real
          val freq_cis_imag: Tensor2D = fullWeights.freq_cis_imag
          val wcls: Tensor2D = d2("output.weight")

          def quantizeQ4: Weights = throw new IllegalArgumentException("Not supported")
          def quantizeQ8: Weights = throw new IllegalArgumentException("Not supported")
        }

    (config, vocab, weights)
  }
}