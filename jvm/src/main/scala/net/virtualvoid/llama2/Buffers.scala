package net.virtualvoid.llama2

import java.io.{ File, RandomAccessFile }
import java.nio.ByteOrder
import java.nio.channels.FileChannel

trait Buffers {
  def next(size: Long): FloatBuffer
}

object Buffers {
  def fromFile(file: File, skipHeader: Int): Buffers = new Buffers {
    // memory map the file and setup the weight buffers
    def mapBuffer(): FloatBuffer = {
      val f = new RandomAccessFile(file, "r")
      val buffer = f.getChannel.map(FileChannel.MapMode.READ_ONLY, 0, f.length)
      buffer.order(ByteOrder.LITTLE_ENDIAN)
      buffer.position(skipHeader)
      buffer.asFloatBuffer()
    }
    val floatBuffer = mapBuffer()

    def next(sizeL: Long): FloatBuffer = {
      require(sizeL <= Int.MaxValue)
      val size = sizeL.toInt
      val res = floatBuffer.slice()
      res.limit(size)
      floatBuffer.position(floatBuffer.position() + size)
      res
    }
  }
}