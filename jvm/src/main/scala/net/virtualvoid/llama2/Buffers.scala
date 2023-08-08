package net.virtualvoid.llama2

import java.io.{ File, RandomAccessFile }
import java.nio.ByteOrder
import java.nio.channels.FileChannel

trait Buffers {
  def d1(dim1: Int): Tensor1D
  def d2(dim1: Int, dim2: Int): Tensor2D
  def d3(dim1: Int, dim2: Int, dim3: Int): Tensor3D
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
    def d2(dim1: Int, dim2: Int): Tensor2D = Tensor2D(next(dim1 * dim2), dim1, dim2)
    def d3(dim1: Int, dim2: Int, dim3: Int): Tensor3D = {
      import scala.collection.parallel.CollectionConverters._
      val elements =
        (0 until dim1)
          .map(_ => d2(dim2, dim3))
      //.par.map(_.quantizeQ8).seq

      new Tensor3D {
        override def size0: Int = dim1
        override def size1: Int = dim2
        override def size2: Int = dim3

        override def apply(i: Int): Tensor2D = elements(i)

        override def toFloatArray: Array[Float] = ???
        override def toFloatBuffer: FloatBuffer = ???
      }
    }
  }
}