package net.virtualvoid.llama2

import java.io.{ File, RandomAccessFile }
import java.nio.ByteOrder
import java.nio.channels.FileChannel

import scalanative.posix.sys.mman

trait Buffers {
  def d1(dim1: Int): Tensor1D
  def d2(dim1: Int, dim2: Int): Tensor2D
  def d3(dim1: Int, dim2: Int, dim3: Int): Tensor3D
}

object Buffers {
  def fromFile(file: File, skipHeader: Int): Buffers = new Buffers {

    import scalanative.unsafe._
    import scalanative.unsigned._
    import scalanative.posix.fcntl._

    /** position in the file in f32 units */
    var pos = 7L // skip header
    val buf: Ptr[Byte] =
      Zone { implicit z =>
        val fd = open(toCString(file.getPath()), O_RDONLY, 0.toUInt)
        mman.mmap(null, file.length().toULong, mman.PROT_READ, mman.MAP_SHARED, fd, 0)
      }

    def next(size: Long): FloatBuffer = {
      val res = FloatBuffer(buf.asInstanceOf[Ptr[Float]], pos, size.toInt)
      pos += size
      res
    }

    def d1(dim1: Int): Tensor1D = Tensor1D(next(dim1), dim1)
    def d2(dim1: Int, dim2: Int): Tensor2D = Tensor2D(next(dim1 * dim2), dim1, dim2)
    def d3(dim1: Int, dim2: Int, dim3: Int): Tensor3D = Tensor3D(next(dim1 * dim2 * dim3), dim1, dim2, dim3)
  }
}