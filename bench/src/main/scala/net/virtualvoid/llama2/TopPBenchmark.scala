package net.virtualvoid.llama2

import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.io.FileInputStream
import java.util
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPInputStream
import scala.io.Source
import scala.util.Random

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

  @Param(Array("histogram", "quick-select-top-p", "quick-find-max-top-p", "filterAndScan", "filterAndFastSort", "filterAndSort", "sorting"))
  var method: String = _

  private var impl: TopPSampling = _
  private var data: Seq[Array[Float]] = _

  val p = 0.9f

  @Setup
  def setup(): Unit = {
    impl = method match {
      case "sorting"              => NaiveSortingTopP
      case "filterAndSort"        => FilterAndSort
      case "filterAndFastSort"    => FilterAndFastSort
      case "filterAndScan"        => FilterAndScan
      case "quick-find-max-top-p" => QuickFindMaxTopP
      case "quick-select-top-p"   => QuickSelectTopP
      case "histogram"            => HistogramSearch
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
          throw new IllegalStateException("Result differed, see above")
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
  def println(str: String): Unit = ()

  def indices(vs: Array[Float], p: Float): Array[Int] = {
    // idea: Finding top_p elements is equivalent to finding the separating element between the top_p and the rest.
    //       If we build a histogram in a way that models the distribution well enough, we might be able to find
    //       a good guess for the separating element in a single pass and only have to figure out the details out
    //       of a small set considering only the bin at the separating edge.
    val cutoff = (1f - p) / (vs.length - 1)

    val histo: Array[Int] = new Array[Int](200) // FIXME: figure out required size
    val sum: Array[Float] = new Array[Float](200)

    // first pass: calculate histogram
    {
      var i = 0
      while (i < vs.length) {
        val v = vs(i)
        if (v >= cutoff) {
          val bin = (-math.log(v) * 3f).toInt
          histo(bin) += 1
          sum(bin) += v
        }
        i += 1
      }
    }

    // print histo
    {
      var i = 0
      var cumsum = 0f
      while (i < histo.length) {
        cumsum += sum(i)
        println(f"$i%5d ${histo(i)}%5d ${sum(i)}%12.9f $cumsum%12.9f")
        i += 1
      }
    }

    var cdf = 0f
    var lastBucket = 0
    var numElements = 0

    while (lastBucket < histo.length && cdf < p) {
      numElements += histo(lastBucket)
      cdf += sum(lastBucket)
      lastBucket += 1
    }
    // select toCheck bucket
    lastBucket -= 1

    println(f"lastBucket: $lastBucket numElements: $numElements cdf: $cdf")

    // all in lastBucket - 1
    val selectedThreshold = math.exp(-(lastBucket - 1 + 1) / 3f)
    val toCheckThreshold = math.exp(-(lastBucket + 1) / 3f)
    println(f"selectedThreshold: $selectedThreshold%12.9f toCheckThreshold: $toCheckThreshold%12.9f")

    val result = new Array[Int](numElements)
    var selected = 0
    var toCheck = numElements - histo(lastBucket)
    println(f"selected: $selected toCheck: $toCheck")

    // second pass: collect elements
    {
      var i = 0
      while (i < vs.length) {
        val v = vs(i)
        if (v > selectedThreshold) {
          result(selected) = i
          selected += 1
        } else if (v > toCheckThreshold) {
          result(toCheck) = i
          toCheck += 1
        }

        i += 1
      }
    }
    require(selected == numElements - histo(lastBucket), f"selected: $selected numElements: $numElements histo(lastBucket): ${histo(lastBucket)}")

    val remainingCdf = p - (cdf - sum(lastBucket))

    val sorted = result.drop(selected).take(toCheck - selected).sortBy(-vs(_))
    val cumSum = sorted.iterator.scanLeft(0f)(_ + vs(_))
    val idx = cumSum.indexWhere(_ > remainingCdf)
    val selectedIdxs = sorted.take(idx)

    result.take(selected) ++ selectedIdxs
  }
}

/**
 * This is a variant where filtering is applied in each step. It seems to be much slower, maybe
 * because the main loop is more complicated and has more complicated access patterns since it
 * keeps moving elements around till the end.
 */
object QuickFindMaxTopP extends TopPSampling {
  def indices(vs: Array[Float], p: Float): Array[Int] = {
    //strategy:
    // * in each iteration, simultaneously
    //   * find the largest value and move it to the front
    //   * filter in all values that still can be part of the result and move them to front
    // after each iteration the working array:
    //   - contains the i largest values sorted in the beginning
    //   - then a number of values that still need to be regarded for the result
    //   - the tail end, does not have to be stored, but rather the last index we are still interested in

    val idxs = vs.indices.toArray
    var selected = 0
    var numInteresting = vs.length
    // running sum of selected elements
    var cdf = 0f
    // running sum of deselected elements
    var removed = 0f

    while (cdf < p) {
      var max = 0f
      var maxIdx = -1
      val remaining = 1f - cdf - removed // remaining probability mass
      val halfRemaining = remaining / 2f
      val cutoff = (1f - p - removed) / (numInteresting - selected - 1)

      var i = selected
      while (i < numInteresting && max < halfRemaining) {
        val v = vs(idxs(i))
        if (v > max) {
          max = v
          maxIdx = i

          i += 1
        } else if (v < cutoff) {
          numInteresting -= 1
          val tmp = idxs(i)
          idxs(i) = idxs(numInteresting)
          idxs(numInteresting) = tmp
          removed += v

          // don't increase i
        } else
          i += 1
      }

      // swap max with selected
      val tmp = idxs(selected)
      idxs(selected) = idxs(maxIdx)
      idxs(maxIdx) = tmp
      selected += 1
      cdf += max

    }

    idxs.take(selected)
  }
}

/**
 * Use a variant of quickselect to find the top_p elements.
 *
 * The idea is to maintain three areas:
 *  * the top elements < p
 *  * the bottom elements < (1 - p)
 *  * the interesting elements in the middle
 *
 * Then, find a pivot, swap the elements around separated by the pivot value,
 * and figure out, if we can move either the top elements to the already selected group (if sum + current cdf < p),
 * or whether we can discard the bottom elements.
 *
 * FIXME:
 *  * One problem is that the selection of the pivot gets tricky, when only few elements are left, and you cannot make
 *    any progress on some pivot choices. Quickselect originally uses random pivots, which works but might be too expensive.
 * Maybe just cancelling the algorithm when we cannot make progress and the number of remaining elements is low enough,
 * to go back to scanning might be the right choice here.
 *  * The current problem seems to be that the algorithm seems more difficult to terminate (than the classic top-k version)?
 *  * Or maybe something is just wrong with the tricky indexing logic in edge cases.
 */
object QuickSelectTopP extends TopPSampling {
  def indices(vs: Array[Float], p: Float): Array[Int] = {
    //strategy:
    // a refinement of QuickFindMaxTopP that avoids sorting
    // * in each iteration
    //  * choose pivot
    //  * select remaining values in three groups:
    //   - greater than pivot
    //   - smaller than cutoff
    //   - smaller than pivot

    val idxs = vs.indices.toArray
    var selected = 0
    var numInteresting = vs.length
    // running sum of selected elements
    var cdf = 0f
    // running sum of deselected elements
    var removed = 0f

    var count = 0
    val r = new Random(0)

    def println(str: String): Unit = ()

    while (cdf < p) {
      val remaining = 1f - cdf - removed // remaining probability mass
      //val halfRemaining = remaining / 2f
      val cutoff = (1f - p - removed) / (numInteresting - selected - 1)

      var left = selected
      var leftSum = 0f
      var rightSum = 0f
      var right = numInteresting - 1

      // how to choose the pivot?
      //(1 - cdf - removed) * 0.1f // middle of remaining probability mass
      val pivotIdx = left + r.nextInt().abs % (right - left + 1) //(left + right) / 2
      val pivotV = vs(idxs(pivotIdx))

      import Console.{ println => _, _ }
      val nums = idxs.map(vs).map(_ formatted "%8.6f").take(5)
      println(s"$GREEN${nums.take(selected).mkString(", ")}$RESET | $YELLOW${nums.take(numInteresting).drop(left).mkString(", ")}$RESET | $RED${nums.drop(numInteresting).mkString(", ")}$RESET")

      println(f"selected: $selected%5d cdf: $cdf%12.9f removed: $removed%12.9f numInteresting: $numInteresting%5d cutoff: $cutoff%12.9f pivotV: $pivotV%12.9f")

      while (left < right) {
        while ( /*left < right &&*/ vs(idxs(left)) >= pivotV) {
          println(f"left: $left%5d v: ${vs(idxs(left))}%12.9f >= $pivotV%12.9f")
          leftSum += vs(idxs(left))
          left += 1
        }
        while ( /*left < right &&*/ vs(idxs(right)) < pivotV) {
          println(f"right: $right%5d v: ${vs(idxs(right))}%12.9f < $pivotV%12.9f")
          rightSum += vs(idxs(right))
          right -= 1
        }

        if (left < right) {
          println(f"swapping $left%5d ${vs(idxs(left))}%12.9f and $right%5d ${vs(idxs(right))}%12.9f")
          val tmp = idxs(left)
          idxs(left) = idxs(right)
          idxs(right) = tmp

          leftSum += vs(idxs(left))
          rightSum += vs(idxs(right))

          left += 1
          right -= 1
        }
      }
      leftSum += vs(idxs(left))
      if (cdf + leftSum < p) {
        // all elements in left are selected, great, we did a bunch of work
        cdf += leftSum
        selected = left
      } else {
        // great we can discard all elements in right and go on with the rest on the left
        numInteresting = right + 1 // TODO: off by one?
        removed += rightSum
      }
      println(f"left: $left%5d leftSum: $leftSum%12.9f right: $right%5d rightSum: $rightSum%12.9f")
      if (numInteresting - selected == 1) {
        cdf += vs(idxs(selected))
        selected += 1
      }

      count += 1
      if (count == 40) ???
    }

    idxs.take(selected)
  }
}