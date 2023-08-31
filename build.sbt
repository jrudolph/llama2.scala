import scala.scalanative.build._

val scalaV = "3.3.0"
val pekkoV = "1.0.1"
val pekkoHttpV = "1.0.0"
val scalaTestV = "3.2.16"

lazy val llama2 =
  crossProject(JVMPlatform, NativePlatform)
    .crossType(CrossType.Full)
    .in(file("."))
    .settings(
      Seq(
        libraryDependencies ++= Seq(
          "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4",
          "org.scalatest" %% "scalatest" % scalaTestV % Test,
          "org.scalatestplus" %% "scalacheck-1-17" % (scalaTestV + ".0") % Test
        ),
        scalaVersion := scalaV
      )
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

lazy val bench = project.in(file("bench"))
  .dependsOn(llama2.jvm)
  .enablePlugins(JmhPlugin)
  .settings(
    scalaVersion := scalaV,
    libraryDependencies += "io.spray" %% "spray-json" % "1.3.6"
  )

lazy val web = project.in(file("web"))
  .dependsOn(llama2.jvm)
  .enablePlugins(SbtTwirl)
  .settings(
    scalaVersion := scalaV,
    libraryDependencies ++=
      Seq(
        "org.apache.pekko" %% "pekko-stream" % pekkoV,
        "org.apache.pekko" %% "pekko-http" % pekkoHttpV,
        "org.apache.pekko" %% "pekko-http-caching" % pekkoHttpV,
        "io.spray" %% "spray-json" % "1.3.6",
        "com.typesafe.play" %% "twirl-api" % "1.6.0-RC4"
      ),

    // setup docker build
    // use separate dependency and app jars
    assembly / assemblyOption := (assembly / assemblyOption).value.withIncludeScala(false).withIncludeDependency(false),
    assembly / assemblyJarName := "app.jar", // contract with Dockerfile
    assemblyPackageDependency / assemblyJarName := "deps.jar" // contract with Dockerfile
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