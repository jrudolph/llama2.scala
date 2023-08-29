package net.virtualvoid.llama2.web

import org.apache.pekko.http.scaladsl.marshalling.{ Marshaller, ToEntityMarshaller }
import org.apache.pekko.http.scaladsl.model.MediaTypes._
import org.apache.pekko.http.scaladsl.model.MediaType
import play.twirl.api.{ Xml, Txt, Html }

// copied from https://github.com/btomala/akka-http-twirl/blob/master/src/main/scala/akkahttptwirl/TwirlSupport.scala
object TwirlSupport extends TwirlSupport

trait TwirlSupport {

  /** Serialize Twirl `Html` to `text/html`. */
  val twirlHtmlMarshaller = twirlMarshaller[Html](`text/html`)
  implicit val twirlHtmlMarshallerWrapInPage: ToEntityMarshaller[Html] =
    Marshaller.StringMarshaller.wrap(`text/html`)(c => html.page(c).toString)

  /** Serialize Twirl `Txt` to `text/plain`. */
  implicit val twirlTxtMarshaller: ToEntityMarshaller[Txt] = twirlMarshaller[Txt](`text/plain`)

  /** Serialize Twirl `Xml` to `text/xml`. */
  implicit val twirlXmlMarshaller: ToEntityMarshaller[Xml] = twirlMarshaller[Xml](`text/xml`)

  /** Serialize Twirl formats to `String`. */
  protected def twirlMarshaller[A <: AnyRef: Manifest](contentType: MediaType): ToEntityMarshaller[A] =
    Marshaller.StringMarshaller.wrap(contentType)(_.toString)

}