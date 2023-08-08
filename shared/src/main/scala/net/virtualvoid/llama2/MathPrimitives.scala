package net.virtualvoid.llama2

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

  val Q8_K = 32

  /** Quantizes the given buffer into byte values and factor in blocks of 32 elements */
  def quantizeQ8(floatBuffer: FloatBuffer, size0: Int, size1: Int): (Array[Byte], Array[Float]) = {
    val K = Q8_K
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

  /** Quantizes the given buffer into byte values and factor in blocks of 32 elements */
  def quantizeQ8(floats: Array[Float], destV: Array[Byte], destF: Array[Float]): Unit = {
    val K = Q8_K
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
    val K = Q8_K
    val numBlocks = dim2 / 32
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
}
