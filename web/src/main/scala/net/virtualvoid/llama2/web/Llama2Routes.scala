package net.virtualvoid.llama2
package web

import org.apache.pekko.http.scaladsl.model.{ StatusCodes, Uri }
import org.apache.pekko.http.scaladsl.server.Directives

class Llama2Routes(initialState: Llama2State) extends Directives with TwirlSupport {
  val transformer = Llama2TensorTransformer.init(initialState.model)
  val sampler = TemperatureSampling(1f, 0.99f)

  val initTokens = initialState.next(30, transformer, sampler)

  val main =
    concat(
      path("prompt") {
        parameter("tokens".as[Seq[Int]]) { toks =>
          val state = initialState.prompt(toks)
          complete(html.state(state))
        }
      },
      redirect(Uri(s"/prompt?tokens=${initTokens.stateHistory.map(_.chosenToken).mkString(",")}"), StatusCodes.Found)
    )
}
