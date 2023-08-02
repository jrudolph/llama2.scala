package net.virtualvoid.llama2

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
  def ∘=(other: Tensor1D): Unit

  def ∘(other: Tensor1D): Op1D

  def expMut(): Unit

  def :=(op: Op1D): Unit
  def :=(orig: Tensor1D): Unit

  /** Returns a new tensor backed by the same data but assuming a different dimension */
  def shorten(newDim: Int): Tensor1D

  def toFloatArray: Array[Float]
  def toFloatBuffer: FloatBuffer

  def copyToArray(dest: Array[Float], offset: Int = 0): Unit
}
object Tensor1D {
  def zero(dim1: Int): Tensor1D = Tensor1D(new Array[Float](dim1), dim1)
  def apply(floatBuffer: FloatBuffer, dim1: Int): Tensor1D = new Tensor1D { thisTensor =>
    def size: Int = dim1

    def max: Float = ???
    def sum: Float = ???

    def *(other: Tensor1D): Float = ???
    def +=(other: Tensor1D): Unit = ???
    def +=(scalar: Float): Unit = ???
    def *=(scalar: Float): Unit = ???
    def ∘=(other: Tensor1D): Unit = ???

    def expMut(): Unit = ???

    def :=(op: Op1D): Unit = ???
    def :=(orig: Tensor1D): Unit = ???

    def ∘(other: Tensor1D): Op1D = new Op1D {
      def into(dest: Tensor1D): Unit = {
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

    def shorten(newDim: Int): Tensor1D = ???

    def toFloatArray: Array[Float] = ???
    def toFloatBuffer: FloatBuffer = floatBuffer

    def copyToArray(dest: Array[Float], offset: Int): Unit =
      floatBuffer.duplicate().get(dest, offset, dim1)
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

    def ∘=(other: Tensor1D): Unit = {
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

    def ∘(other: Tensor1D): Op1D = ???

    def shorten(newDim: Int): Tensor1D = {
      require(newDim <= size)
      Tensor1D(fs, newDim, offset)
    }

    def toFloatArray: Array[Float] =
      if (offset == 0 && fs.length == dim) fs
      else ???

    def toFloatBuffer: FloatBuffer = ???

    def copyToArray(dest: Array[Float], destOffset: Int): Unit =
      System.arraycopy(fs, offset, dest, destOffset, dim)
  }

  implicit def autoBuffer(t1: Tensor1D): FloatBuffer = t1.toFloatBuffer
  implicit def autoArray(t1: Tensor1D): Array[Float] = t1.toFloatArray
}

trait Op1D {
  def into(dest: Tensor1D): Unit
}

trait Tensor2D {
  def size0: Int
  def size1: Int

  def apply(i: Int): Tensor1D

  def `@`(v: Tensor1D): Op1D

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

    override def `@`(v: Tensor1D): Op1D = new Op1D {
      override def into(dest: Tensor1D): Unit = {
        require(v.size == dim2)
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
        }
      }
    }

    def toFloatArray: Array[Float] = ???
    def toFloatBuffer: FloatBuffer = floatBuffer.duplicate()
  }

  def apply(fs: Array[Float], dim1: Int, dim2: Int, offset: Int = 0): Tensor2D = new Tensor2D {
    require(fs.size >= offset + dim1 * dim2)
    def floats(i: Int): Float = fs(offset + i)

    def size0: Int = dim1
    def size1: Int = dim2

    def apply(i: Int): Tensor1D = Tensor1D(fs, dim2, offset = offset + i * dim2)
    def `@`(v: Tensor1D): Op1D = ???

    def toFloatArray: Array[Float] = if (offset == 0 && fs.length == dim1 * dim2) fs else ???
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

    def apply(i: Int): Tensor2D = Tensor2D(floats, dim2, dim3, offset = i * dim2 * dim3)

    def toFloatArray: Array[Float] = floats
    def toFloatBuffer: FloatBuffer = ???
  }
  implicit def autoBuffer(t3: Tensor3D): FloatBuffer = t3.toFloatBuffer
  implicit def autoArray(t3: Tensor3D): Array[Float] = t3.toFloatArray
}