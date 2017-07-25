package com.knoldus.mlib

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println

import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
// $example on$
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
// $example off$

object MLibApp{

  def main(args: Array[String]): Unit = {

    //creating sparkSession
    val spark = SparkSession.builder().master("local").appName("mlib").getOrCreate()
    val sc = spark.sparkContext//new SparkContext(conf)
    sc.setLogLevel("ERROR")
    //deleting the previous stored data
    FileUtils.deleteDirectory(new File("target/tmp/myDecisionTreeClassificationModel"))
    // $example on$
    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "src/main/resources/a6a.t")
    // Split the data into training and test sets (30% held out for testing)
    val splits: Array[RDD[LabeledPoint]] = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "entropy"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)
//    implicit val evidence = scala.reflect.ClassTag[(Double, Double)]
    // Evaluate model on test instances and compute test error
    val labelAndPreds: RDD[(Double, Double)] = testData.map[(Double, Double)] { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model:\n" + model.toDebugString)

    // Save and load model
    model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
    val sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")
    // $example off$

    sc.stop()
  }
}
// scalastyle:on println
