package net.virtualvoid.llama2.web

import java.awt.Color

object Colors {

  trait Palette {
    thisPalette =>
    def colorRgbFor(f: Float): Int

    def contains(f: Float): Boolean

    def colorFor(f: Float): String = {
      val c = new Color(colorRgbFor(f))
      val r = c.getRed
      val g = c.getGreen
      val b = c.getBlue
      f"#$r%02x$g%02x$b%02x"
    }
    def colorHSVFor(f: Float): Seq[Float] = {
      val c = new Color(colorRgbFor(f))
      Color.RGBtoHSB(c.getRed, c.getGreen, c.getBlue, null).toSeq
    }

    def |(other: Palette): Palette = new Palette {
      override def colorRgbFor(f: Float): Int =
        if (thisPalette.contains(f)) thisPalette.colorRgbFor(f)
        else other.colorRgbFor(f)

      override def contains(f: Float): Boolean = thisPalette.contains(f) || other.contains(f)
    }

    def comap(f: Float => Float): Palette = new Palette {
      override def colorRgbFor(v: Float): Int = thisPalette.colorRgbFor(f(v))

      override def contains(v: Float): Boolean = thisPalette.contains(f(v))
    }
  }

  val GreenToRed =
    linearPalette(1, 0, Seq(100f / 360f, 1f, 0.6f), Seq(0f, 1f, 1f)).comap(x => x)

  val WhiteToBlue =
    linearPalette(1, 0, Seq(190f / 360f, 0f, 1f), Seq(190f / 360f, 1f, 1f)).comap(x => x * x)

  val AttentionColoring =
    linearPalette(0, 1, Seq(190f / 360f, 0f, 1f), Seq(190f / 360f, 1f, 1f))

  val Classifier =
    linearPalette(-1000, -50, Seq(0f, 1f, 1f), Seq(0f, 1f, 1f)) |
      linearPalette(-50, 0, Seq(0f, 1f, 1f), Seq(0f, 0f, 1f)) |
      linearPalette(0, 50, Seq(100f / 360f, 0f, 1f), Seq(100f / 360f, 1f, 1f)) |
      linearPalette(50, 1000, Seq(100f / 360f, 1f, 1f), Seq(100f / 360f, 1f, 1f))

  val ClassifierMatch = Classifier.comap(x => x * 30f)

  def logarithmicPalette(min: Float, max: Float, fromHsv: Seq[Float], toHsv: Seq[Float]): Palette =
    linearPalette(math.log(min).toFloat, math.log(max).toFloat, fromHsv, toHsv).comap(math.log(_).toFloat)

  def linearPalette(min: Float, max: Float, from: Color, to: Color): Palette =
    linearPalette(min, max, hsvOf(from), hsvOf(to))

  def linearPalette(min: Float, max: Float, fromHsv: Seq[Float], toHsv: Seq[Float]): Palette = {
    val hueI = linearInterpolator(min, max, fromHsv(0), toHsv(0))
    val satI = linearInterpolator(min, max, fromHsv(1), toHsv(1))
    val valI = linearInterpolator(min, max, fromHsv(2), toHsv(2))
    palette(min, max, hueI, satI, valI)
  }

  def linearInterpolator(min: Float, max: Float, rangeMin: Float, rangeMax: Float): Float => Float = {
    val domainSpan = max - min
    val rangeSpan = rangeMax - rangeMin

    v => {
      val r0 = rangeMin + (v - min) * rangeSpan / domainSpan
      val r =
        if (rangeMax > rangeMin)
          r0 min rangeMax max rangeMin
        else
          r0 max rangeMax min rangeMin
      r
    }
  }

  def palette(min: Float, max: Float,
              hInterpolator: Float => Float,
              sInterpolator: Float => Float,
              vInterpolator: Float => Float): Palette =
    new Palette {
      override def colorRgbFor(f: Float): Int = {
        //println(s"for $f ${hueI(f)} ${satI(f)} ${valI(f)}")
        Color.HSBtoRGB(hInterpolator(f), sInterpolator(f), vInterpolator(f))
      }

      override def contains(f: Float): Boolean = min <= f && f <= max
    }

  def hueOf(color: Color): Float = hsvOf(color)(0)
  def hsvOf(color: Color): Seq[Float] = {
    val hsv = new Array[Float](3)
    Color.RGBtoHSB(color.getRed, color.getGreen, color.getBlue, hsv)
    hsv.toIndexedSeq
  }
}
