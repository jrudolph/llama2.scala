package net.virtualvoid.llama2

import java.io.FileInputStream
import java.util.zip.GZIPInputStream
import scala.io.Source
import scala.util.Random

/** Run statistics on gathered probability stats */
object PropStats extends App {
  //  val rand = new Random(3)
  //  val floats = Array.fill(15)(rand.nextFloat() * 10)
  //  Tensor1DMut(floats, floats.size).softmaxMut()
  //  QuickSelectTopP.indices(floats, 0.9f)
  //  ???

  import scala.collection.parallel.CollectionConverters._
  val numElements = 10000
  val data =
    Source.fromInputStream(new GZIPInputStream(new FileInputStream("../outprobs.txt.gz"))).getLines()
      .drop(947)
      .take(1)
      //.drop(46).take(1)
      //.take(1)
      .toVector
      .par
      .map(_.split(",").map(_.toFloat))
      .toVector

  val p = 0.9f

  trait Stat {
    def name: String
    def calc(vector: Array[Float]): Float
  }
  def stat(_name: String, f: Array[Float] => Float) = new Stat {
    def name = _name
    def calc(vector: Array[Float]) = f(vector)
  }

  val Min = stat("min", _.min)
  val Max = stat("max", _.max)
  val Count = stat("count", _.size)
  val Median = stat("median", vector => {
    val sorted = vector.sorted
    val mid = sorted.size / 2
    sorted(mid)
  })

  val Stats = Seq(Min, Max, Count, Median)

  def result(vector: Array[Float], p: Float): Array[Float] =
    FilterAndScan.indices(vector, p).map(i => vector(i))

  val results = data.map(result(_, p))

  // calculate distribution of stat over all data elements
  def dist(stat: Stat): Seq[Float] =
    results.map(stat.calc).sorted

  /*def printDist(stat: Stat): Unit = {
    val d = dist(stat)
    val histo = d.groupBy(v => (math.log(v) * 3f).toInt).view.mapValues(_.size).toVector.sortBy(_._1)
    println(s"${stat.name} distribution:")
    histo.foreach {
      case (log, count) =>
        val v = math.exp(log / 3f)
        println(f"<= $v%8.6f: $count")
    }
  }
  printDist(Min)
  printDist(Count)*/

  val worstIdx = results.zipWithIndex.maxBy(_._1.size)._2
  //println(s"Worst: $worstIdx size: ${results(worstIdx).size}")

  val cutoff = (1 - p) / (data.head.size - 1)
  println(f"Cutoff: $cutoff%12.9f")
  val min = results(worstIdx).min - cutoff
  println(f"Last element: $min%12.9f")
  data(worstIdx).groupBy(v => (-math.log(v - cutoff) * 3f).toInt).view.mapValues(_.size).toVector.sortBy(_._1).foreach {
    case (log, count) =>
      val v = math.exp(-log / 3f)
      if (v > min && math.exp((log - 1) / 3f) < min)
        println(f"Last element: $min%12.9f")
      println(f"<= $v%12.9f (log $log%3d): $count")
  }

  val res = FilterAndScan.indices(data(worstIdx), p).map(i => data(worstIdx)(i)).sorted.reverse
  println("Reference:")
  res.foreach(x => println(f"$x%12.9f"))

  println(s"${res.size} elements")

  val res2 = HistogramSearch.indices(data(worstIdx), p).map(i => data(worstIdx)(i)).sorted.reverse
  println("Hist:")
  res2.foreach(x => println(f"$x%12.9f"))

  println(s"Ref und new equal: ${res.toSet == res2.toSet} refsum: ${res.sum} newsum: ${res2.sum} numref: ${res.size} numnew: ${res2.size}")
}
