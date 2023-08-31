package net.virtualvoid.llama2.web

import com.typesafe.config.Config

case class AppConfig(
    host:  String,
    port:  Int,
    model: String
)

object AppConfig {
  def fromConfig(config: Config): AppConfig = {
    AppConfig(
      host = config.getString("app.host"),
      port = config.getInt("app.port"),
      model = config.getString("app.model")
    )
  }
}