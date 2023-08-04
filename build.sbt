import scala.scalanative.build._

val scalaV = "3.3.0"

lazy val llama2 =
  crossProject(JVMPlatform, NativePlatform)
    .crossType(CrossType.Full)
    .in(file("."))
    .settings(
      scalaVersion := scalaV
    )
    .jvmSettings(
      javacOptions ++= Seq("-h", "c")
    )
    .nativeSettings(
      nativeConfig ~= {
        _.withLTO(LTO.thin)
          .withMode(Mode.releaseFast)
          .withGC(GC.commix)
      }
    )

// docs

enablePlugins(ParadoxMaterialThemePlugin)

paradoxProperties ++= Map(
  "github.base_url" -> (Compile / paradoxMaterialTheme).value.properties.getOrElse("repo", "")
)

Compile / paradoxMaterialTheme := {
  ParadoxMaterialTheme()
    // choose from https://jonas.github.io/paradox-material-theme/getting-started.html#changing-the-color-palette
    .withColor("light-green", "amber")
    // choose from https://jonas.github.io/paradox-material-theme/getting-started.html#adding-a-logo
    .withLogoIcon("cloud")
    .withCopyright("Copyleft © Johannes Rudolph")
    .withRepository(uri("https://github.com/jrudolph/xyz"))
    .withSocial(
      uri("https://github.com/jrudolph"),
      uri("https://twitter.com/virtualvoid")
    )
}