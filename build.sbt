val scalaV = "2.12.6"
val specs2V = "4.3.2"

libraryDependencies ++= Seq(
  "org.specs2" %% "specs2-core" % specs2V % "test"
)

scalaVersion := scalaV
