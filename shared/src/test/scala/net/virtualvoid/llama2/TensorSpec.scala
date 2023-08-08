package net.virtualvoid.llama2

import org.scalacheck.{ Arbitrary, Gen }
import org.scalatest.freespec.AnyFreeSpec
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class TensorSpec extends AnyFreeSpec with ScalaCheckPropertyChecks {
  "Tensor1D" - {
    "can be built from FloatBuffer" in {
      forAll(genFloatBuffer()) { buf =>
        val t = Tensor1D(buf, buf.limit)
        assert(t.size == buf.limit)
        assert(t.toFloatBuffer.array().toSeq == buf.array().toSeq)
      }
    }
    "must support ∘" in {
      forAll(genMatchingTensor1DAndTensor1DMut()) {
        case (t, mut) =>
          val res = Tensor1DMut.zero(t.size)
          res := t ∘ mut

          (0 until t.size).foreach { i =>
            assert(res(i) == t.get(i) * mut(i))
          }
      }
    }
    "must support copyToArray" in {
      forAll(genTensor1D()) { t =>
        val res = new Array[Float](t.size)
        t.copyToArray(res)

        assert(res.toSeq == t.toFloatBuffer.array().toSeq)
      }
    }
  }

  "Tensor1DMut" - {
    "can be built from Array[Float]" in {
      forAll(genFloatArray()) { arr =>
        val t = Tensor1DMut(arr, arr.size)
        assert(t.size == arr.size)
        assert(t.toFloatArray.toSeq == arr.toSeq)
      }
    }
    "support in place operations" - {
      "+=" in {
        forAll(genTensor1DMut(), genFloat) { (t, f) =>
          val tData = t.toFloatArray.clone()
          t += f

          assert(almostEqual(tData.map(_ + f), t.toFloatArray))
        }
      }
      "-=" in {
        forAll(genTensor1DMut(), genFloat) { (t, f) =>
          val tData = t.toFloatArray.clone()
          t -= f

          assert(almostEqual(tData.map(_ - f), t.toFloatArray))
        }
      }
      "*=" in {
        forAll(genTensor1DMut(), genFloat) { (t, f) =>
          val tData = t.toFloatArray.clone()
          t *= f

          assert(almostEqual(tData.map(_ * f), t.toFloatArray))
        }
      }
      "/=" in {
        forAll(genTensor1DMut(), genFloat.filterNot(_.abs < 0.1f)) { (t, f) =>
          val tData = t.toFloatArray.clone()
          t /= f

          assert(almostEqual(tData.map(_ / f), t.toFloatArray))
        }
      }
      "expMut" in {
        forAll(genTensor1DMut()) { t =>
          val tData = t.toFloatArray.clone()
          t.expMut()

          assert(almostEqual(tData.map(Math.exp(_).toFloat), t.toFloatArray))
        }
      }
      "∘=" in pending
    }
    "support reductions" - {
      "max" in {
        forAll(genTensor1DMut()) { t =>
          assert(t.toFloatArray.max == t.max)
        }
      }
      "sum" in {
        forAll(genTensor1DMut()) { t =>
          assert(t.toFloatArray.sum == t.sum)
        }
      }
    }
    "shorten" in pending
    "copyToArray" in pending
    "Tensor1D with offset" in pending
  }

  "Tensor2D" - {
    "can be built from FloatBuffer" in {
      forAll(genFloatBuffer(Gen.choose(0, 100).map(_ * 5))) { buf =>
        val t = Tensor2D(buf, buf.limit / 5, 5)
        assert(t.size0 == buf.limit / 5)
        assert(t.size1 == 5)
        assert(t.toFloatBuffer.array().toSeq == buf.array().toSeq)
      }
    }
    "operations" - {
      "apply(i) (projection)" in {
        forAll(genTensor2D(), Gen.choose(0, 100)) { (t, i) =>
          whenever(i < t.size0) {
            val res = t(i)
            assert(res.size == t.size1)
            assert(res.toFloatBuffer.array().toSeq == t.toFloatBuffer.slice(i * t.size1, t.size1).array().toSeq)
          }
        }
      }
      "@ (matrix multiplication with vector)" in {
        forAll(genMatchingT2DAndT1DMut()) {
          case (t, v) =>
            val res = Tensor1DMut.zero(t.size0)
            res := t `@` v
            assert(res.size == t.size0)
            assert(almostEqual(
              res.toFloatArray,
              t.toFloatBuffer.array().grouped(t.size1).map { row =>
                row.zip(v.toFloatArray).map { case (a, b) => a * b }.sum
              }.toArray
            ))
        }
      }
    }
    "quantize Q8 @ must give similar results" in {
      forAll(genMatchingT2DAndT1DMut(DefaultSize, Gen.choose(0, 10).map(_ * 32 /* Q8 block size */ ))) {
        case (m, v) =>
          val res = Tensor1DMut.zero(m.size0)
          val res2 = Tensor1DMut.zero(m.size0)
          res := m.quantizeQ8 `@` v
          res2 := m `@` v

          assert(almostEqual(
            res.toFloatArray,
            res2.toFloatArray,
            0.01f
          ))
      }
    }
    "quantize Q4 @ must give similar results" in {
      forAll(genMatchingT2DAndT1DMut(DefaultSize, Gen.choose(0, 10).map(_ * 32 /* Q4 block size */ ))) {
        case (m, v) =>
          val res = Tensor1DMut.zero(m.size0)
          val res2 = Tensor1DMut.zero(m.size0)
          res := m.quantizeQ4 `@` v
          res2 := m `@` v

          assert(almostEqual(
            res.toFloatArray,
            res2.toFloatArray,
            0.1f
          ))
      }
    }
  }

  def almostEqual(a: Float, b: Float, eps: Float): Boolean = {
    val diff = Math.abs(a - b)
    diff < eps || diff / a.abs.max(b.abs).max(1) < eps
  }
  def almostEqual(a: Array[Float], b: Array[Float], eps: Float = 0.000001f): Boolean = {
    a.length == b.length && a.zip(b).forall { case (a, b) => almostEqual(a, b, eps) }
  }

  val DefaultSize = 100

  def genMatchingT2DAndT1DMut(dim1: Gen[Int] = DefaultSize, dim2: Gen[Int] = DefaultSize): Gen[(Tensor2D, Tensor1DMut)] =
    for {
      d1 <- dim1; d2 <- dim2
      m <- genTensor2D(d1, d2)
      v <- genTensor1DMut(d2)
    } yield (m, v)

  def genFloat: Gen[Float] = Gen.double.map(_.toFloat)
  def genFloatBuffer(size: Gen[Int] = DefaultSize): Gen[FloatBuffer] =
    genFloatArray(size).map(a => java.nio.FloatBuffer.wrap(a))

  def genTensor1D(size: Gen[Int] = DefaultSize): Gen[Tensor1D] =
    for {
      buf <- genFloatBuffer(size)
    } yield Tensor1D(buf, buf.limit)

  def genTensor2D(dim1: Gen[Int] = DefaultSize, dim2: Gen[Int] = DefaultSize): Gen[Tensor2D] =
    for {
      d1 <- dim1; d2 <- dim2
      buf <- genFloatBuffer(d1 * d2)
    } yield Tensor2D(buf, d1, d2)

  def genMatchingTensor1DAndTensor1DMut(size: Gen[Int] = DefaultSize): Gen[(Tensor1D, Tensor1DMut)] =
    for {
      t <- genTensor1D(size)
      mut <- genTensor1DMut(Gen.const(t.size))
    } yield (t, mut)

  def genTensor1DMut(size: Gen[Int] = DefaultSize): Gen[Tensor1DMut] = genFloatArray(size).map(a => Tensor1DMut(a, a.size))
  def genFloatArray(size: Gen[Int] = DefaultSize): Gen[Array[Float]] = size.flatMap(size => Gen.listOfN(size, genFloat).map(_.toArray))
}