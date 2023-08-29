package net.virtualvoid.llama2.web

import org.apache.pekko.http.scaladsl.server.Directives

class Llama2Routes extends Directives with TwirlSupport {
  val main = complete("Test")
}
