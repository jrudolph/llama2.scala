package net.virtualvoid.llama2
import java.io.File
import java.nio.ByteBuffer

object AVX2MathImplementation extends MathImplementation {
  {
    val libCandidates = Seq("libmatmul.so", "../libmatmul.so", "../c/libmatmul.so").map(new File(_))
    libCandidates.find(_.exists) match {
      case Some(f) => System.load(f.getAbsolutePath)
      case None =>
        println(s"Couldn't find libmatmul.so in any of ${libCandidates.map(_.getAbsolutePath).mkString(", ")} - trying to loading from system path")
        System.loadLibrary("matmul")
    }
  }

  def matMul(m: FloatBuffer, vs: Array[Float], dest: Array[Float]): Unit =
    VectMult.matMul(m, vs, dest)

  def matMulQ4Q8(quantized: Array[Byte], quantizeFactor: Array[Float], quantizedV: Array[Byte], quantizeVFactor: Array[Float], dim1: Int, dim2: Int, dest: Array[Float]): Unit =
    VectMult.matMulQ4(quantized, quantizeFactor, quantizedV, quantizeVFactor, dest)

  def matMulQ4Q8FromBuffer(buffer: ByteBuffer, quantizedV: Array[Byte], quantizeVFactor: Array[Float], dim1: Int, dim2: Int, dest: Array[Float]): Unit =
    VectMult.matMulQ4_buffer(buffer, quantizedV, quantizeVFactor, dest)

  def matMulQ8(quantized: Array[Byte], quantizeFactor: Array[Float], quantizedV: Array[Byte], quantizeVFactor: Array[Float], dim1: Int, dim2: Int, dest: Array[Float]): Unit =
    VectMult.matMulQ8(quantized, quantizeFactor, quantizedV, quantizeVFactor, dest)

  // currently no optimized versions for these
  def quantizeQ4(floatBuffer: FloatBuffer, size0: Int, size1: Int): (Array[Byte], Array[Float]) = ScalaMathImplementation.quantizeQ4(floatBuffer, size0, size1)
  def quantizeQ8(floatBuffer: FloatBuffer, size0: Int, size1: Int): (Array[Byte], Array[Float]) = ScalaMathImplementation.quantizeQ8(floatBuffer, size0, size1)
  def quantizeQ8(floats: Array[Float], destV: Array[Byte], destF: Array[Float]): Unit = ScalaMathImplementation.quantizeQ8(floats, destV, destF)
}
