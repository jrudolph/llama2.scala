package net.virtualvoid.llama2

abstract class MathImplementation {
  def matMul(m: FloatBuffer, vs: Array[Float], dest: Array[Float]): Unit
  def matMulQ4Q8(quantized: Array[Byte], quantizeFactor: Array[Float], quantizedV: Array[Byte], quantizeVFactor: Array[Float], dim1: Int, dim2: Int, dest: Array[Float]): Unit
  def matMulQ4Q8FromBuffer(buffer: java.nio.ByteBuffer, quantizedV: Array[Byte], quantizeVFactor: Array[Float], dim1: Int, dim2: Int, dest: Array[Float]): Unit
  def matMulQ8(quantized: Array[Byte], quantizeFactor: Array[Float], quantizedV: Array[Byte], quantizeVFactor: Array[Float], dim1: Int, dim2: Int, dest: Array[Float]): Unit

  def quantizeQ4(floatBuffer: FloatBuffer, size0: Int, size1: Int): (Array[Byte], Array[Float])
  def quantizeQ8(floatBuffer: FloatBuffer, size0: Int, size1: Int): (Array[Byte], Array[Float])
  def quantizeQ8(floats: Array[Float], destV: Array[Byte], destF: Array[Float]): Unit
}

object MathImplementation {
  private[this] var TheMathImplementation: MathImplementation = ScalaMathImplementation

  def setImplementation(impl: MathImplementation): Unit =
    TheMathImplementation = impl

  def Default: MathImplementation = TheMathImplementation
}