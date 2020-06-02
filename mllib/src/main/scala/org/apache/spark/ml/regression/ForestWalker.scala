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

import scala.collection.mutable.{ArrayBuffer, PriorityQueue}

import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

import org.apache.spark.annotation.Since
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.QuantileRandomForest
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{RandomForestModel => OldRandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._


case class ForestWalker(treeLabels: PriorityQueue[TreeLabelWalker] = PriorityQueue[TreeLabelWalker]())
  extends Iterator[(Double,Double)]
{
  def snarfLabels(sortedLabels: ArrayBuffer[Float], numTrees: Int) : Unit = {
    treeLabels += TreeLabelWalker.make(sortedLabels, numTrees)
  }
  def peek : Double = { treeLabels.head.currValue }
  override def hasNext : Boolean = { ! (treeLabels.isEmpty) }
  override def next() : (Double,Double) = {
    val treeLabelWalker = treeLabels.dequeue
    val label = treeLabelWalker.currValue.toDouble
    val weight = treeLabelWalker.weight
    if ( treeLabelWalker.more ) {
      treeLabelWalker.advance()
      treeLabels += treeLabelWalker
    }
    (label, weight)
  }
}
