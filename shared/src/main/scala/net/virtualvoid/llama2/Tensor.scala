package net.virtualvoid.llama2

import java.nio.ByteOrder

trait Tensor1D {
  def size: Int

  def ∘(other: Tensor1DMut): Op1D
  def toFloatBuffer: FloatBuffer
  def copyToArray(dest: Array[Float], offset: Int = 0): Unit
}
object Tensor1D {
  def apply(floatBuffer: FloatBuffer, dim1: Int): Tensor1D = new Tensor1D {
    thisTensor =>
    def size: Int = dim1

    def ∘(other: Tensor1DMut): Op1D = new Op1D {
      def into(dest: Tensor1DMut): Unit = {
        require(other.size == thisTensor.size)
        val destArray = dest.toFloatArray
        val others = other.toFloatArray
        var i = 0
        while (i < size) {
          destArray(i) = floatBuffer.get(i) * others(i)
          i += 1
        }
      }
    }

    def toFloatBuffer: FloatBuffer = floatBuffer
    def copyToArray(dest: Array[Float], offset: Int): Unit =
      floatBuffer.duplicate().get(dest, offset, dim1)
  }
  implicit def autoBuffer(t1: Tensor1D): FloatBuffer = t1.toFloatBuffer
}

trait Tensor1DMut extends Tensor1D {
  def max: Float
  def sum: Float
  def *(other: Tensor1DMut): Float
  def +=(other: Tensor1DMut): Unit
  def +=(scalar: Float): Unit
  def -=(scalar: Float): Unit = this += -scalar

  def *=(scalar: Float): Unit
  def /=(scalar: Float): Unit = this *= 1f / scalar

  /* element-wise multiplication */
  def ∘=(other: Tensor1DMut): Unit

  def expMut(): Unit

  def :=(op: Op1D): Unit
  def :=(orig: Tensor1D): Unit

  /** Returns a new tensor backed by the same data but assuming a different dimension */
  def shorten(newDim: Int): Tensor1DMut

  def toFloatArray: Array[Float]
}
object Tensor1DMut {
  def zero(dim1: Int): Tensor1DMut = Tensor1DMut(new Array[Float](dim1), dim1)

  def apply(fs: Array[Float], dim: Int, offset: Int = 0): Tensor1DMut = new Tensor1DMut {
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

    def *(other: Tensor1DMut): Float = {
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
    def +=(other: Tensor1DMut): Unit = {
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

    def ∘=(other: Tensor1DMut): Unit = {
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

    def :=(op: Op1D): Unit = op.into(this)
    def :=(orig: Tensor1D): Unit =
      orig.copyToArray(fs, offset)

    def ∘(other: Tensor1DMut): Op1D = ???

    def shorten(newDim: Int): Tensor1DMut = {
      require(newDim <= size)
      Tensor1DMut(fs, newDim, offset)
    }

    def toFloatArray: Array[Float] =
      if (offset == 0 && fs.length == dim) fs
      else ???

    def toFloatBuffer: FloatBuffer = ???

    def copyToArray(dest: Array[Float], destOffset: Int): Unit =
      System.arraycopy(fs, offset, dest, destOffset, dim)
  }

  implicit def autoArray(t1: Tensor1DMut): Array[Float] = t1.toFloatArray
}

/** An operation that still needs a destination to run */
trait Op1D {
  def into(dest: Tensor1DMut): Unit
}

trait Tensor2D {
  def size0: Int
  def size1: Int

  def apply(i: Int): Tensor1D

  def `@`(v: Tensor1DMut): Op1D

  def toFloatArray: Array[Float]
  def toFloatBuffer: FloatBuffer

  def quantizeQ8: Tensor2D
}

object Tensor2D {
  def apply(fb: FloatBuffer, dim1: Int, dim2: Int): Tensor2D = new Tensor2D { outer =>
    val floatBuffer = fb.duplicate()
    def size0: Int = dim1
    def size1: Int = dim2
    def apply(i: Int): Tensor1D = {
      val source = floatBuffer.duplicate().position(i * dim2).slice()
      Tensor1D(source, dim2)
    }

    def `@`(v: Tensor1DMut): Op1D = new Op1D {
      override def into(dest: Tensor1DMut): Unit = {
        require(v.size == dim2)
        require(dest.size == dim1)
        require(v.size % 16 == 0)

        VectMult.matMul(floatBuffer, v, dest)

        /*//require(v.size == dim2)
        //require(dest.size == dim1)
        val vs = v.toFloatArray
        var i = 0
        while (i < dim1) {
          var j = 0
          var sum = 0.0f
          while (j < dim2) {
            sum += floatBuffer.get(i * dim2 + j) * vs(j)
            j += 1
          }
          dest(i) = sum
          i += 1
        }*/
      }
    }

    def toFloatArray: Array[Float] = ???
    def toFloatBuffer: FloatBuffer = floatBuffer.duplicate()

    def quantizeQ8: Tensor2D = {
      val K = 32
      require(size1 % K == 0)

      val numBlocks = size0 * size1 / K

      val quantized = VectMult.allocateAligned(size0 * size1, 32)
      val quantizeFactor = VectMult.allocateAligned(numBlocks * 4, 32).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
      //val quantized = new Array[Byte](size0 * size1)
      //val quantizeFactor = new Array[Float](numBlocks)

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

        quantizeFactor.put(i, d)

        j = 0
        while (j < K) {
          val v = floatBuffer.get(i * K + j)
          val x0 = v * id // scale
          quantized.put(i * K + j, math.round(x0).toByte)
          //println(f"At ${i * K + j} v: $v%.6f x0: ${x0} quantized: ${quantized(i * K + j)}")
          j += 1
        }

        i += 1
      }

      new Tensor2D {
        def size0: Int = dim1
        def size1: Int = dim2

        def apply(i: Int): Tensor1D = ???

        def `@`(v: Tensor1DMut): Op1D = { dest =>
          val arr = v.toFloatArray
          require(arr.length % K == 0)

          val numBlocksV = arr.size / K

          // quantize v as well (it will be used `dim1` times so it is worth it)
          val quantizedV = VectMult.allocateAligned(arr.length, 32)
          val quantizeVFactor = VectMult.allocateAligned(numBlocksV * 4, 32).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

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

              quantizeVFactor.put(i, d)
              j = 0
              while (j < K) {
                val x0 = arr(i * K + j) * id // scale
                quantizedV.put(i * K + j, math.round(x0).toByte)
                j += 1
              }

              i += 1
            }
          }

          //val dest2 = new Array[Float](dim1)
          val res = VectMult.matMulQ8New(quantized, quantizeFactor, quantizedV, quantizeVFactor, dest, dim1, dim2)

          VectMult.freeAligned(quantizedV)
          VectMult.freeAligned(quantizeVFactor)

          res
          /*var i = 0
          while (i < dim1) {
            var j = 0
            var sum = 0.0f
            require(numBlocksV * 32 == dim2)
            while (j < numBlocksV) {
              var sumq = 0
              var k = 0
              while (k < K) {
                sumq += quantized(i * dim2 + j * K + k) * quantizedV(j * K + k)
                //if (i == 0 && j == 0)
                //  println(f"i: $i j: $j k: $k sumq: $sumq")
                k += 1
              }
              sum += sumq.toFloat * quantizeFactor(i * dim2 / K + j) * quantizeVFactor(j)

              j += 1
            }

            dest(i) = sum
            i += 1
          }
          dest.toFloatArray.zip(dest2).take(10).foreach(println)
          ???*/

        }

        def toFloatArray: Array[Float] = ???
        def toFloatBuffer: FloatBuffer = ???

        def quantizeQ8: Tensor2D = this
      }
    }
  }

  implicit def autoBuffer(t2: Tensor2D): FloatBuffer = t2.toFloatBuffer
  implicit def autoArray(t2: Tensor2D): Array[Float] = t2.toFloatArray
}

trait Tensor2DMut extends Tensor2D {
  def apply(i: Int): Tensor1DMut
}
object Tensor2DMut {
  def zero(dim1: Int, dim2: Int): Tensor2DMut = Tensor2DMut(new Array[Float](dim1 * dim2), dim1, dim2)

  def apply(fs: Array[Float], dim1: Int, dim2: Int, offset: Int = 0): Tensor2DMut = new Tensor2DMut {
    require(fs.size >= offset + dim1 * dim2)

    def floats(i: Int): Float = fs(offset + i)
    def size0: Int = dim1
    def size1: Int = dim2

    def apply(i: Int): Tensor1DMut = Tensor1DMut(fs, dim2, offset = offset + i * dim2)

    def `@`(v: Tensor1DMut): Op1D = ???

    def toFloatArray: Array[Float] = if (offset == 0 && fs.length == dim1 * dim2) fs else ???
    def toFloatBuffer: FloatBuffer = ???
    def quantizeQ8: Tensor2D = ???
  }
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

  implicit def autoBuffer(t3: Tensor3D): FloatBuffer = t3.toFloatBuffer
  implicit def autoArray(t3: Tensor3D): Array[Float] = t3.toFloatArray
}

trait Tensor3DMut extends Tensor3D {
  def apply(i: Int): Tensor2DMut
}
object Tensor3DMut {
  def zero(dim1: Int, dim2: Int, dim3: Int): Tensor3DMut = Tensor3DMut(new Array[Float](dim1 * dim2 * dim3), dim1, dim2, dim3)

  def apply(floats: Array[Float], dim1: Int, dim2: Int, dim3: Int): Tensor3DMut = new Tensor3DMut {
    require(floats.size == dim1 * dim2 * dim3)

    def size0: Int = dim1
    def size1: Int = dim2
    def size2: Int = dim3

    def apply(i: Int): Tensor2DMut = Tensor2DMut(floats, dim2, dim3, offset = i * dim2 * dim3)
    def toFloatArray: Array[Float] = floats
    def toFloatBuffer: FloatBuffer = ???

    def quantize: Tensor3D = ???
  }
}