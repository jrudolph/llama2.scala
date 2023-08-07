package net.virtualvoid.llama2

trait Tensor1D[d1 <: Int] {
  def size: d1

  def ∘(other: Tensor1DMut[d1]): Op1D[d1]
  def toFloatBuffer: FloatBuffer
  def copyToArray(dest: Array[Float], offset: Int = 0): Unit
}
object Tensor1D {
  def apply(floatBuffer: FloatBuffer, dim1: Int): Tensor1D[dim1.type] = new Tensor1D[dim1.type] {
    thisTensor =>
    def size: dim1.type = dim1

    def ∘(other: Tensor1DMut[dim1.type]): Op1D[dim1.type] = new Op1D[dim1.type] {
      def into(dest: Tensor1DMut[dim1.type]): Unit = {
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
  implicit def autoBuffer(t1: Tensor1D[_]): FloatBuffer = t1.toFloatBuffer
}

trait Tensor1DMut[d1 <: Int] extends Tensor1D[d1] {
  def max: Float
  def sum: Float
  def *(other: Tensor1DMut[d1]): Float
  def +=(other: Tensor1DMut[d1]): Unit
  def +=(scalar: Float): Unit
  def -=(scalar: Float): Unit = this += -scalar

  def *=(scalar: Float): Unit
  def /=(scalar: Float): Unit = this *= 1f / scalar

  /* element-wise multiplication */
  def ∘=(other: Tensor1DMut[d1]): Unit

  def expMut(): Unit

  def :=(op: Op1D[d1]): Unit
  def :=(orig: Tensor1D[d1]): Unit

  /** Returns a new tensor backed by the same data but assuming a different dimension */
  def shorten(newDim: Int): Tensor1DMut[newDim.type]

  def toFloatArray: Array[Float]
}
object Tensor1DMut {
  def zero(dim1: Int): Tensor1DMut[dim1.type] = Tensor1DMut(new Array[Float](dim1), dim1)

  def apply(fs: Array[Float], dim: Int, offset: Int = 0): Tensor1DMut[dim.type] = new Tensor1DMut[dim.type] {
    require(fs.size >= offset + dim)

    def floats(i: Int): Float = fs(offset + i)

    def size: dim.type = dim

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

    def *(other: Tensor1DMut[dim.type]): Float = {
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
    def +=(other: Tensor1DMut[dim.type]): Unit = {
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

    def ∘=(other: Tensor1DMut[dim.type]): Unit = {
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

    def :=(op: Op1D[dim.type]): Unit = op.into(this)
    def :=(orig: Tensor1D[dim.type]): Unit =
      orig.copyToArray(fs, offset)

    def ∘(other: Tensor1DMut[dim.type]): Op1D[dim.type] = ???

    def shorten(newDim: Int): Tensor1DMut[newDim.type] = {
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

  implicit def autoArray(t1: Tensor1DMut[_]): Array[Float] = t1.toFloatArray
}

/** An operation that still needs a destination to run */
trait Op1D[d1 <: Int] {
  def into(dest: Tensor1DMut[d1]): Unit
}

trait Tensor2D[d1 <: Int, d2 <: Int] {
  def size0: d1
  def size1: d2

  def apply(i: Int): Tensor1D[d2]

  def `@`(v: Tensor1DMut[d2]): Op1D[d1]

  def toFloatArray: Array[Float]
  def toFloatBuffer: FloatBuffer
}

object Tensor2D {
  def apply(fb: FloatBuffer, dim1: Int, dim2: Int): Tensor2D[dim1.type, dim2.type] = new Tensor2D[dim1.type, dim2.type] {
    val floatBuffer = fb.duplicate()
    def size0: dim1.type = dim1
    def size1: dim2.type = dim2
    def apply(i: Int): Tensor1D[dim2.type] = {
      val source = floatBuffer.duplicate().position(i * dim2).slice()
      Tensor1D(source, dim2)
    }

    def `@`(v: Tensor1DMut[dim2.type]): Op1D[dim1.type] = new Op1D[dim1.type] {
      override def into(dest: Tensor1DMut[dim1.type]): Unit = {
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

  implicit def autoBuffer(t2: Tensor2D[_, _]): FloatBuffer = t2.toFloatBuffer
  implicit def autoArray(t2: Tensor2D[_, _]): Array[Float] = t2.toFloatArray
}

trait Tensor2DMut[d1 <: Int, d2 <: Int] extends Tensor2D[d1, d2] {
  def apply(i: Int): Tensor1DMut[d2]
}
object Tensor2DMut {
  def zero(dim1: Int, dim2: Int): Tensor2DMut[dim1.type, dim2.type] = Tensor2DMut(new Array[Float](dim1 * dim2), dim1, dim2)

  def apply(fs: Array[Float], dim1: Int, dim2: Int, offset: Int = 0): Tensor2DMut[dim1.type, dim2.type] = new Tensor2DMut[dim1.type, dim2.type] {
    require(fs.size >= offset + dim1 * dim2)

    def floats(i: Int): Float = fs(offset + i)
    def size0: dim1.type = dim1
    def size1: dim2.type = dim2

    def apply(i: Int): Tensor1DMut[dim2.type] = Tensor1DMut(fs, dim2, offset = offset + i * dim2)

    def `@`(v: Tensor1DMut[dim2.type]): Op1D[dim1.type] = ???

    def toFloatArray: Array[Float] = if (offset == 0 && fs.length == dim1 * dim2) fs else ???
    def toFloatBuffer: FloatBuffer = ???
  }
}
trait Tensor3D[d1 <: Int, d2 <: Int, d3 <: Int] {
  def size0: d1

  def size1: d2
  def size2: d3

  def apply(i: Int): Tensor2D[d2, d3]

  def toFloatArray: Array[Float]
  def toFloatBuffer: FloatBuffer
}
object Tensor3D {
  def apply(floatBuffer: FloatBuffer, dim1: Int, dim2: Int, dim3: Int): Tensor3D[dim1.type, dim2.type, dim3.type] = new Tensor3D[dim1.type, dim2.type, dim3.type] {
    def size0: dim1.type = dim1
    def size1: dim2.type = dim2
    def size2: dim3.type = dim3

    def apply(i: Int): Tensor2D[dim2.type, dim3.type] = {
      val source = floatBuffer.duplicate().position(i * dim2 * dim3).slice()
      Tensor2D(source, dim2, dim3)
    }

    def toFloatArray: Array[Float] = ???
    def toFloatBuffer: FloatBuffer = floatBuffer.duplicate()
  }

  implicit def autoBuffer(t3: Tensor3D[_, _, _]): FloatBuffer = t3.toFloatBuffer
  implicit def autoArray(t3: Tensor3D[_, _, _]): Array[Float] = t3.toFloatArray
}

trait Tensor3DMut[d1 <: Int, d2 <: Int, d3 <: Int] extends Tensor3D[d1, d2, d3] {
  def apply(i: Int): Tensor2DMut[d2, d3]
}
object Tensor3DMut {
  def zero(dim1: Int, dim2: Int, dim3: Int): Tensor3DMut[dim1.type, dim2.type, dim3.type] = Tensor3DMut(new Array[Float](dim1 * dim2 * dim3), dim1, dim2, dim3)

  def apply(floats: Array[Float], dim1: Int, dim2: Int, dim3: Int): Tensor3DMut[dim1.type, dim2.type, dim3.type] = new Tensor3DMut[dim1.type, dim2.type, dim3.type] {
    require(floats.size == dim1 * dim2 * dim3)

    def size0: dim1.type = dim1
    def size1: dim2.type = dim2
    def size2: dim3.type = dim3

    def apply(i: Int): Tensor2DMut[dim2.type, dim3.type] = Tensor2DMut(floats, dim2, dim3, offset = i * dim2 * dim3)
    def toFloatArray: Array[Float] = floats
    def toFloatBuffer: FloatBuffer = ???
  }
}