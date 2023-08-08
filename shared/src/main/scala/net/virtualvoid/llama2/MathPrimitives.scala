package net.virtualvoid.llama2

import java.nio.ByteOrder

object MathPrimitives {
  def matMul(m: FloatBuffer, vs: Array[Float], dest: Array[Float]): Unit = {
    val dim2 = vs.length
    val dim1 = dest.length
    var i = 0
    while (i < dim1) {
      var j = 0
      var sum = 0.0f
      while (j < dim2) {
        sum += m.get(i * dim2 + j) * vs(j)
        j += 1
      }
      dest(i) = sum
      i += 1
    }
  }

  val QK4_0 = 32
  val QK8_0 = 32

  def quantizeQ4(floatBuffer: FloatBuffer, size0: Int, size1: Int): (Array[Byte], Array[Float]) = {
    val K = QK4_0
    val numBlocks = size0 * size1 / K

    val quantized = new Array[Byte](size0 * size1 / 2)
    val quantizeFactor = new Array[Float](numBlocks)

    var i = 0
    while (i < numBlocks) {
      var j = 0
      var amax = 0f
      var max = 0f
      while (j < K) {
        val v = floatBuffer.get(i * K + j).abs
        if (v > amax) {
          amax = v
          max = v
        }
        j += 1
      }

      val d = max / -8f
      val id = if (d != 0f) 1.0f / d else 0.0f

      quantizeFactor(i) = d

      j = 0
      while (j < K / 2) {
        val x0 = floatBuffer.get(i * K + j) * id // scale
        val x1 = floatBuffer.get(i * K + K / 2 + j) * id // scale

        val xi0 = (x0 + 8.5f).toByte.min(15)
        val xi1 = (x1 + 8.5f).toByte.min(15)

        quantized(i * K / 2 + j) = ((xi0 << 0) | (xi1 << 4)).toByte

        j += 1
      }

      i += 1
    }

    (quantized, quantizeFactor)
  }

  def matMulQ4Q8(quantized: Array[Byte], quantizeFactor: Array[Float], quantizedV: Array[Byte], quantizeVFactor: Array[Float], dim1: Int, dim2: Int, dest: Array[Float]): Unit = {
    val K = QK4_0
    val numBlocks = dim2 / 32

    var i = 0
    while (i < dim1) {
      var j = 0
      var sum = 0.0f
      while (j < numBlocks) {
        val xF = quantizeFactor(i * dim2 / K + j)
        val vF = quantizeVFactor(j)

        var sumq = 0
        var k = 0
        while (k < K / 2) {
          val idx = i * dim2 / 2 + j * K / 2 + k
          val e = quantized(idx)
          val x0 = (e & 0x0f) - 8
          val x1 = ((e & 0xf0) >> 4) - 8

          val v0 = quantizedV(j * K + k)
          val v1 = quantizedV(j * K + k + K / 2)

          sumq += (x0 * v0) + (x1 * v1)

          k += 1

        }

        sum += sumq.toFloat * xF * vF

        j += 1
      }

      dest(i) = sum
      i += 1
    }
  }

  /**
   * Same as matMulQ4 but accesses model (matrix) weights from packed buffer with GGML Q4_0 blocks:
   *
   * typedef struct {
   *   ggml_fp16_t d;          // delta
   *   uint8_t qs[QK4_0 / 2];  // nibbles / quants
   * } block_q4_0;
   */
  def matMulQ4Q8FromBuffer(buffer: java.nio.ByteBuffer, quantizedV: Array[Byte], quantizeVFactor: Array[Float], dim1: Int, dim2: Int, dest: Array[Float]): Unit = {
    val K = QK4_0
    val numBlocks = dim2 / K
    val shortBuffer = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()

    def quantized(block: Int, offset: Int): Byte =
      buffer.get(block * 18 + 2 /* d */ + offset)

    def quantizeFactor(idx: Int): Float = {
      val bufIdx = idx * 18 / 2
      val fp16d = shortBuffer.get(bufIdx)
      fp16Tofp32(fp16d)
    }

    var i = 0
    while (i < dim1) {
      var j = 0
      var sum = 0.0f
      val blockOffset = i * dim2 / K
      while (j < numBlocks) {
        val blockId = blockOffset + j
        val xF = quantizeFactor(blockId)
        val vF = quantizeVFactor(j)

        var sumq = 0
        var k = 0
        while (k < K / 2) {
          val e = quantized(blockId, k)
          val x0 = (e & 0x0f) - 8
          val x1 = ((e & 0xf0) >> 4) - 8

          val v0 = quantizedV(j * K + k)
          val v1 = quantizedV(j * K + k + K / 2)

          sumq += (x0 * v0) + (x1 * v1)
          k += 1
        }

        sum += sumq.toFloat * xF * vF

        j += 1
      }

      dest(i) = sum
      i += 1
    }
  }

  /** Quantizes the given buffer into byte values and factor in blocks of 32 elements */
  def quantizeQ8(floatBuffer: FloatBuffer, size0: Int, size1: Int): (Array[Byte], Array[Float]) = {
    val K = QK8_0
    require(size1 % K == 0)

    val numBlocks = size0 * size1 / K

    val quantized = new Array[Byte](size0 * size1)
    val quantizeFactor = new Array[Float](numBlocks)

    var i = 0
    while (i < numBlocks) {
      //println(s"Block ${i}")
      var j = 0
      var max = 0f
      while (j < K) {
        val v = floatBuffer.get(i * K + j).abs
        if (v > max) max = v
        j += 1
      }

      val d = max / ((1 << 7) - 1)
      val id = if (d != 0f) 1.0f / d else 0.0f

      quantizeFactor(i) = d

      j = 0
      while (j < K) {
        val v = floatBuffer.get(i * K + j)
        val x0 = v * id // scale
        quantized(i * K + j) = math.round(x0).toByte
        //println(f"At ${i * K + j} v: $v%.6f x0: ${x0} quantized: ${quantized(i * K + j)}")
        j += 1
      }

      i += 1
    }

    (quantized, quantizeFactor)
  }

  def quantizeQ8(floats: Array[Float], destV: Array[Byte], destF: Array[Float]): Unit = {
    val K = QK8_0
    require(floats.length % K == 0)

    val numBlocksV = floats.length / K

    var i = 0
    while (i < numBlocksV) {
      var j = 0
      var max = 0f
      while (j < K) {
        val v = floats(i * K + j).abs
        if (v > max) max = v
        j += 1
      }

      val d = max / ((1 << 7) - 1)
      val id = if (d != 0f) 1.0f / d else 0.0f

      destF(i) = d
      j = 0
      while (j < K) {
        val x0 = floats(i * K + j) * id // scale
        destV(i * K + j) = math.round(x0).toByte
        j += 1
      }

      i += 1
    }
  }

  def matMulQ8(quantized: Array[Byte], quantizeFactor: Array[Float], quantizedV: Array[Byte], quantizeVFactor: Array[Float], dim1: Int, dim2: Int, dest: Array[Float]): Unit = {
    val K = QK8_0
    val numBlocks = dim2 / K
    var i = 0
    while (i < dim1) {
      var j = 0
      var sum = 0.0f
      while (j < numBlocks) {
        var sumq = 0
        var k = 0
        while (k < K) {
          sumq += quantized(i * dim2 + j * K + k) * quantizedV(j * K + k)
          k += 1
        }
        sum += sumq.toFloat * quantizeFactor(i * dim2 / K + j) * quantizeVFactor(j)

        j += 1
      }

      dest(i) = sum
      i += 1
    }
  }

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
}
