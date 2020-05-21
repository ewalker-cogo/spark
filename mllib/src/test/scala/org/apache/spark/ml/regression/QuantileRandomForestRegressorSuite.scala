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

package org.apache.spark.ml.regression

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tree.impl.TreeTests
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTest, MLTestingUtils}
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.{EnsembleTestHelper, RandomForest => OldRandomForest}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, Dataset}

/**
 * Test suite for [[QuantileRandomForestRegressor]].
 */
class QuantileRandomForestRegressorSuite extends MLTest with DefaultReadWriteTest{

  import testImplicits._

  private var orderedLabeledPoints50_1000: RDD[LabeledPoint] = _

  override def beforeAll() {
    super.beforeAll()
    orderedLabeledPoints50_1000 =
      sc.parallelize(EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures = 50, 1000)
        .map(_.asML))
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////
  def testSinglePrediction(model: QuantileRandomForestRegressionModel,
    dataset: Dataset[_]): Unit = {
    var observationsCol = "observations"
    model.getObservations(dataset, observationsCol).select(model.getFeaturesCol, observationsCol)
      .collect().foreach {
      case Row(features: Vector, prediction: Seq[Seq[Float]]) => {
        assert(prediction === model.observations(features))
        //println(prediction.toString) 
      }
    }
    observationsCol = "observations2"
    model.setPredictionCol(observationsCol)
    model.transform(dataset).select(model.getFeaturesCol, observationsCol)
      .collect().foreach {
      case Row(features: Vector, prediction: Seq[Seq[Float]]) => {
        assert(prediction === model.observations(features))
        //println(prediction.toString) 
      }
    }
  }

  test("prediction on single instance") {
    val rf = new QuantileRandomForestRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(10)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)

    val df = orderedLabeledPoints50_1000.toDF()
    val model = rf.fit(df)
    testSinglePrediction(model, df)
  }

}

