package net.virtualvoid.llama2
package web

import org.apache.pekko.http.scaladsl.model.{ StatusCodes, Uri }
import org.apache.pekko.http.scaladsl.server.{ Directive1, Directives, Route }

class Llama2Routes(initialState: Llama2State) extends Directives with TwirlSupport {
  val transformer = Llama2TensorTransformer.init(initialState.model)
  val sampler = TemperatureSampling(1f, 0.99f)

  val initTokens = initialState.next(30, transformer, sampler)

  val main =
    concat(
      path("prompt") {
        stateAtPrompt { state0 =>
          val state = state0.next(transformer, sampler)
          complete(html.state(state))
        }
      },
      path("generate") {
        parameter("method", "n".as[Int]) { (method, n) =>
          val sampler = method match {
            case "topp"   => TemperatureSampling(1f, 0.99f)
            case "argmax" => ArgmaxSampler
          }
          stateAtPrompt { state0 =>
            val state = state0.next(n, transformer, sampler)
            redirectToState(state)
          }
        }
      },
      pathSingleSlash {
        redirectToState(initTokens)
      },
      getFromResourceDirectory("web")
    )

  def stateAtPrompt: Directive1[Llama2State] =
    parameter("tokens".as[Seq[Int]]).flatMap { toks =>
      provide(initialState.prompt(toks))
    }

  def redirectToState(state: Llama2State): Route =
    redirect(Uri(s"/prompt?tokens=${state.stateHistory.map(_.chosenToken).mkString(",")}"), StatusCodes.Found)
}
