package net.virtualvoid.llama2

import java.io.{ File, RandomAccessFile }
import java.nio.ByteOrder
import java.nio.channels.FileChannel

import scalanative.posix.sys.mman

trait Buffers {
  def next(size: Long): FloatBuffer
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
      val res = FloatBuffer(buf.asInstanceOf[Ptr[Float]], pos)
      pos += size
      res
    }
  }
}