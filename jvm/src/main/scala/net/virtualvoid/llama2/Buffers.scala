package net.virtualvoid.llama2

import java.io.{ File, RandomAccessFile }
import java.nio.ByteOrder
import java.nio.channels.FileChannel

trait Buffers {
  def d1(dim1: Int): Tensor1D
  def d2(dim1: Int, dim2: Int): Tensor2D[dim1.type, dim2.type]
  def d3(dim1: Int, dim2: Int, dim3: Int): Tensor3D[dim1.type, dim2.type, dim3.type]
}

object Buffers {
  def fromFile(file: File, skipHeader: Int): Buffers = new Buffers {
    val fch = new RandomAccessFile(file, "r").getChannel

    var pos = skipHeader.toLong
    def next(size: Long): FloatBuffer = {
      val buffer = fch.map(FileChannel.MapMode.READ_ONLY, pos, size * 4)
      buffer.order(ByteOrder.LITTLE_ENDIAN)
      pos += size * 4
      buffer.asFloatBuffer()
    }

    def d1(dim1: Int): Tensor1D = Tensor1D(next(dim1), dim1)
    def d2(dim1: Int, dim2: Int): Tensor2D[dim1.type, dim2.type] = Tensor2D(next(dim1 * dim2), dim1, dim2)
    def d3(dim1: Int, dim2: Int, dim3: Int): Tensor3D[dim1.type, dim2.type, dim3.type] = {
      val elements = for (i <- 0 until dim1) yield d2(dim2, dim3)

      new Tensor3D {
        override def size0: dim1.type = dim1
        override def size1: dim2.type = dim2
        override def size2: dim3.type = dim3

        override def apply(i: Int): Tensor2D[dim2.type, dim3.type] = elements(i)

        override def toFloatArray: Array[Float] = ???
        override def toFloatBuffer: FloatBuffer = ???
      }
    }
  }
}