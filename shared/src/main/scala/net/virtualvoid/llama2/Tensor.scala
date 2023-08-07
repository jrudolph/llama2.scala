package net.virtualvoid.llama2

enum Dim {
  def shape: Seq[Long] = this match {
    case D1(d1) => Seq(d1)
    case D2(d1, d2) => Seq(d1, d2)
    case D3(d1, d2, d3) => Seq(d1, d2, d3)
    case D4(d1, d2, d3, d4) => Seq(d1, d2, d3, d4)
  }

  case D1[D1 <: Int](d1: D1)
  case D2[D1 <: Int, D2 <: Int](d1: D1, d2: D2)
  case D3[D1 <: Int, D2 <: Int, D3 <: Int](d1: D1, d2: D2, d3: D3)
  case D4[D1 <: Int, D2 <: Int, D3 <: Int, D4 <: Int](d1: D1, d2: D2, d3: D3, d4: D4)
}
object Dim {
  type Join[dA <: Dim, dB <: Dim] <: Dim = (dA, dB) match {
    case (D1[a], D1[b]) => D2[a, b]
    case (D1[a], D2[b, c]) => D3[a, b, c]
    case (D2[a, b], D1[c]) => D3[a, b, c]
    case (D2[a, b], D2[c, d]) => D4[a, b, c, d]
    case (D1[a], D3[b, c, d]) => D4[a, b, c, d]
    case (D3[a, b, c], D1[d]) => D4[a, b, c, d]
  }
  type Drop1[D <: Dim] <: Dim = D match {
    case D2[a, b] => D1[b]
    case D3[a, b, c] => D2[b,c]
    case D4[a, b, c, d] => D3[ b, c, d]
  }

  def apply(d1: Int): D1[d1.type] = D1(d1)
  def apply(d1: Int, d2: Int): D2[d1.type, d2.type] = D2(d1, d2)
}

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
}

object Tensor2D {
  def apply(fb: FloatBuffer, dim1: Int, dim2: Int): Tensor2D = new Tensor2D {
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
  }
}
trait Tensor3D[d1 <: Int, d2 <: Int, d3 <: Int] {
  def size0: d1

  def size1: d2
  def size2: d3

  def apply(i: Int): Tensor2D

  def toFloatArray: Array[Float]
  def toFloatBuffer: FloatBuffer
}
object Tensor3D {
  def apply(floatBuffer: FloatBuffer, dim1: Int, dim2: Int, dim3: Int): Tensor3D[dim1.type, dim2.type, dim3.type] = new Tensor3D[dim1.type, dim2.type, dim3.type] {
    def size0: dim1.type = dim1
    def size1: dim2.type = dim2
    def size2: dim3.type = dim3

    def apply(i: Int): Tensor2D = {
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
  def apply(i: Int): Tensor2DMut
}
object Tensor3DMut {
  def zero(dim1: Int, dim2: Int, dim3: Int): Tensor3DMut[dim1.type, dim2.type, dim3.type] = Tensor3DMut(new Array[Float](dim1 * dim2 * dim3), dim1, dim2, dim3)

  def apply(floats: Array[Float], dim1: Int, dim2: Int, dim3: Int): Tensor3DMut[dim1.type, dim2.type, dim3.type] = new Tensor3DMut[dim1.type, dim2.type, dim3.type] {
    require(floats.size == dim1 * dim2 * dim3)

    def size0: dim1.type = dim1
    def size1: dim2.type = dim2
    def size2: dim3.type = dim3

    def apply(i: Int): Tensor2DMut = Tensor2DMut(floats, dim2, dim3, offset = i * dim2 * dim3)
    def toFloatArray: Array[Float] = floats
    def toFloatBuffer: FloatBuffer = ???
  }
}