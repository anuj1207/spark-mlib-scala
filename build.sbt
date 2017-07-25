name := "spark-mlib-sample"

version := "1.0"

scalaVersion := "2.11.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.1.0",
  "org.apache.spark" %% "spark-mllib" % "2.1.0" //% "provided"
)