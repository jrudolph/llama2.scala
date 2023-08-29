package net.virtualvoid.llama2.web

import org.apache.pekko.actor.ActorSystem
import org.apache.pekko.http.scaladsl.Http

import scala.util.{ Failure, Success }

object Llama2WebMain extends App {
  implicit val system: ActorSystem = ActorSystem()

  import system.dispatcher

  val appConfig = AppConfig.fromConfig(system.settings.config)
  val routes = new Llama2Routes().main

  val server = Http().newServerAt(appConfig.host, appConfig.port).bind(routes)
  server.onComplete {
    case Success(s) =>
      println(s"Server started on http:/${s.localAddress}")
    case Failure(ex) =>
      println(s"Server could not be started: $ex")
      system.terminate()
  }
}
