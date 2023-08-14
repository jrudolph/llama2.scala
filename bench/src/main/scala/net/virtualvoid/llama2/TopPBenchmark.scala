package net.virtualvoid.llama2

import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.io.FileInputStream
import java.util
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPInputStream
import scala.io.Source

@State(Scope.Thread)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(
  value = 3,
  jvmArgs = Array(
    "-server",
    "-Xms1g",
    "-Xmx1g",
    "-XX:NewSize=500m",
    "-XX:MaxNewSize=500m",
    "-XX:InitialCodeCacheSize=512m",
    "-XX:ReservedCodeCacheSize=512m",
    "-XX:+AlwaysPreTouch"))
@BenchmarkMode(Array(Mode.Throughput))
@OutputTimeUnit(TimeUnit.SECONDS)
class TopPBenchmark {

  @Param(Array("filterAndScan", "filterAndFastSort", "filterAndSort", "sorting"))
  var method: String = _

  private var impl: TopPSampling = _
  private var data: Seq[Array[Float]] = _

  val p = 0.9f

  @Setup
  def setup(): Unit = {
    impl = method match {
      case "sorting"           => NaiveSortingTopP
      case "filterAndSort"     => FilterAndSort
      case "filterAndFastSort" => FilterAndFastSort
      case "filterAndScan"     => FilterAndScan
    }

    data =
      Source.fromInputStream(new GZIPInputStream(new FileInputStream("../outprobs.txt.gz"))).getLines()
        .take(100)
        .map(_.split(",").map(_.toFloat))
        .toVector

    data.zipWithIndex.foreach {
      case (d, i) =>
        val res = impl.indices(d, p).toSet
        val ref = NaiveSortingTopP.indices(d, p).toSet

        if (res != ref) {
          println(s"Result for $i differs")
          println(s"impl selected: ${res.size} ref selected: ${ref.size}")
          println(s"impl sum: ${res.map(d(_)).sum} ref sum: ${ref.map(d(_)).sum}")
          println(s"Ref ranking")
          ref.toVector.sortBy(-d(_)).zipWithIndex.foreach {
            case (idx, i) =>
              println(f"  $i%5d $idx%5d ${d(idx)}%12.9f")
          }
          throw new IllegalStateException
        }
    }
  }

  @Benchmark
  @OperationsPerInvocation(100)
  def topP(blackhole: Blackhole): Unit = {
    data.foreach { vs =>
      blackhole.consume(impl.indices(vs, p))
    }
  }
}

trait TopPSampling {
  def indices(vs: Array[Float], p: Float): Array[Int]
}

object NaiveSortingTopP extends TopPSampling {
  def indices(vs: Array[Float], p: Float): Array[Int] = {
    val sorted = vs.zipWithIndex.sortBy(-_._1).toVector
    val cumSum = sorted.iterator.scanLeft(0f)(_ + _._1)
    val idx = cumSum.indexWhere(_ > p)
    sorted.take(idx).map(_._2).toArray
  }
}

object FilterAndSort extends TopPSampling {
  def indices(vs: Array[Float], p: Float): Array[Int] = {
    val cutoff = (1 - p) / (vs.length - 1)
    val sorted = vs.zipWithIndex.filter(_._1 >= cutoff).sortBy(-_._1).toVector
    val cumSum = sorted.iterator.scanLeft(0f)(_ + _._1)
    val idx = cumSum.indexWhere(_ > p)
    sorted.take(idx).map(_._2).toArray
  }
}

object FilterAndFastSort extends TopPSampling {
  def indices(vs: Array[Float], p: Float): Array[Int] = {
    val cutoff = (1 - p) / (vs.length - 1)
    val idxs = new Array[Integer](vs.length)
    var selected = 0

    {
      var i = 0
      while (i < idxs.length) {
        val v = vs(i)
        if (v >= cutoff) {
          idxs(selected) = i
          selected += 1
        }
        i += 1
      }
    }
    util.Arrays.sort(idxs, 0, selected, (a: Integer, b: Integer) => java.lang.Float.compare(vs(b), vs(a)))

    val cumSum = idxs.iterator.take(selected).scanLeft(0f)((sum, i) => sum + vs(i))
    val idx = cumSum.indexWhere(_ > p)
    idxs.take(idx).map(_.intValue)
  }
}

object FilterAndScan extends TopPSampling {
  def indices(vs: Array[Float], p: Float): Array[Int] = {
    // values smaller than (1 - topp) / (logits.length - 1) cannot be part of the result, so sort them out directly
    val cutoff = (1f - p) / (vs.length - 1)

    val idxs = new Array[Int](vs.length)
    var selected = 0

    {
      var i = 0
      while (i < idxs.length) {
        val v = vs(i)
        if (v >= cutoff) {
          idxs(selected) = i
          selected += 1
        }
        i += 1
      }
    }

    def maxLessThan(remaining: Float, prevMax: Float, prevIndex: Int): Int = {
      val halfRemaining = remaining / 2
      var maxIndex = -1
      var max = -1f
      var i = 0
      while (i < selected && max < halfRemaining) {
        val idx = idxs(i)
        val v = vs(idx)
        if (v > max && ((v < prevMax) || (v == prevMax && idx > prevIndex))) {
          max = v
          maxIndex = idx
        }
        i += 1
      }
      maxIndex
    }

    val scanAttempts = 1000
    val idxBuffer = new Array[Int](scanAttempts)
    val pBuffer = new Array[Float](scanAttempts)

    def collect(numFound: Int, sum: Float, prevMax: Float, prevIndex: Int): Int =
      if (sum > p) numFound
      else if (numFound >= idxBuffer.size) throw new IllegalStateException(s"After $numFound iterations sum is $sum")
      else {
        val maxIdx = maxLessThan(1f - sum, prevMax, prevIndex)
        val v = vs(maxIdx)
        idxBuffer(numFound) = maxIdx
        pBuffer(numFound) = v
        collect(numFound + 1, sum + v, v, maxIdx)
      }

    val numFound = collect(0, 0f, 2f, -1)
    if (numFound == -1) throw new IllegalStateException("Too many values")
    else idxBuffer.take(numFound)
  }
}

object HistogramSearch extends TopPSampling {
  def indices(vs: Array[Float], p: Float): Array[Int] = {
    // idea: Finding top_p elements is equivalent to finding the separating element between the top_p and the rest.
    //       If we build a histogram in a way that models the distribution well enough, we might be able to find
    //       a good guess for the separating element in a single pass and only have to figure out the details out
    //       of a small set considering only the bin at the separating edge.


    ???
  }
}

object QuickTopP extends TopPSampling {
  def indices(vs: Array[Float], p: Float): Array[Int] = {
    //strategy:
    // * in each iteration, simultaneously
    //   * find the largest value and move it to the front
    //   * filter out all values that cannot be part of the result and move them back
    // after each iteration the working array:
    //   - contains the i largest values sorted in the beginning
    //   - then a number of values that still need to be regarding for the result
    //   - the tail end, does not have to be stored, but rather the last index we are still interested in

    // values smaller than (1 - topp) / (logits.length - 1) cannot be part of the result, so sort them out directly
    val cutoff = (1f - p) / (vs.length - 1)

    val idxs = new Array[Int](vs.length)
    var selected = 0

    {
      var i = 0
      while (i < idxs.length) {
        val v = vs(i)
        if (v >= cutoff) {
          idxs(selected) = i
          selected += 1
        }
        i += 1
      }
    }

    def maxLessThan(remaining: Float, prevMax: Float, prevIndex: Int): Int = {
      val halfRemaining = remaining / 2
      var maxIndex = -1
      var max = -1f
      var i = 0
      while (i < selected && max < halfRemaining) {
        val idx = idxs(i)
        val v = vs(idx)
        if (v > max && ((v < prevMax) || (v == prevMax && idx > prevIndex))) {
          max = v
          maxIndex = idx
        }
        i += 1
      }
      maxIndex
    }

    val scanAttempts = 1000
    val idxBuffer = new Array[Int](scanAttempts)
    val pBuffer = new Array[Float](scanAttempts)

    def collect(numFound: Int, sum: Float, prevMax: Float, prevIndex: Int): Int =
      if (sum > p) numFound
      else if (numFound >= idxBuffer.size) throw new IllegalStateException(s"After $numFound iterations sum is $sum")
      else {
        val maxIdx = maxLessThan(1f - sum, prevMax, prevIndex)
        val v = vs(maxIdx)
        idxBuffer(numFound) = maxIdx
        pBuffer(numFound) = v
        collect(numFound + 1, sum + v, v, maxIdx)
      }

    val numFound = collect(0, 0f, 2f, -1)
    if (numFound == -1) throw new IllegalStateException("Too many values")
    else idxBuffer.take(numFound)
  }
}