package net.virtualvoid.llama2
package web

import org.apache.pekko.actor.ActorSystem
import org.apache.pekko.http.scaladsl.Http

import java.io.File
import scala.util.{ Failure, Success }

object Llama2WebMain extends App {
  implicit val system: ActorSystem = ActorSystem()

  import system.dispatcher

  val baseDir = { // reStart runs with a subdirectory as the working directory
    val firstTry = new File("tokenizer.bin")
    if (firstTry.exists()) firstTry.getParentFile
    else new File("..")
  }
  //val checkpointFile = new File(baseDir, "llama2_7b.bin")
  val checkpointFile = new File(baseDir, "stories15M.bin")
  //val checkpointFile = new File(baseDir, "stories42M.bin")
  //val checkpointFile = new File(baseDir, "stories110M.bin")
  val tokenizerFile = new File(baseDir, "tokenizer.bin")

  val ggmlFile = new File(baseDir, "llama-2-7b.ggmlv3.q4_0.bin")

  val model = Llama2Model.fromLlama2CModel(checkpointFile, tokenizerFile)
  //val model = Llama2Model.fromGgml(ggmlFile)

  def printConfig(config: Config): Unit = {
    def p(name: String, value: Int): Unit = println(f"$name%-20s: $value%5d")

    p("Layers", config.nLayers)
    p("Dimension", config.dim)
    p("Hidden Dimension", config.hiddenDim)
    p("Heads", config.nHeads)
    p("Head Size", config.headSize)
    p("Vocab Size", config.vocabSize)
  }

  printConfig(model.config)
  val initial = Llama2State(model)
  AVX2MathImplementation
  VectMult.setParallelism(6)

  val appConfig = AppConfig.fromConfig(system.settings.config)
  val routes = new Llama2Routes(initial).main

  val server = Http().newServerAt(appConfig.host, appConfig.port).bind(routes)
  server.onComplete {
    case Success(s) =>
      println(s"Server started on http:/${s.localAddress}")
    case Failure(ex) =>
      println(s"Server could not be started: $ex")
      system.terminate()
  }
}
