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

package org.apache.spark.ml.tree

import scala.collection.mutable.{ArrayBuffer}

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator
import org.apache.spark.mllib.tree.model.{ImpurityStats, InformationGainStats => OldInformationGainStats, Node => OldNode, Predict => OldPredict}

/**
 * Decision tree leaf node.
 * @param prediction  Prediction this node makes
 * @param impurity  Impurity measure at this node (for training data)
 */
class LeafNodeV2 private[ml] (
    prediction: Double,
    impurity: Double,
    impurityStats: ImpurityCalculator,
    sortedLabels: ArrayBuffer[Float]) extends LeafNode(prediction, impurity, impurityStats) {

  def getSortedLabels(): ArrayBuffer[Float] = { sortedLabels }

  override def toString: String =
    s"LeafNode(prediction = $prediction, impurity = $impurity)"

  override private[ml] def predictImpl(features: Vector): LeafNodeV2 = this

  override private[tree] def numDescendants: Int = 0

  override private[tree] def subtreeToString(indentFactor: Int = 0): String = {
    val prefix: String = " " * indentFactor
    prefix + s"Predict: $prediction\n"
  }

  override private[tree] def subtreeDepth: Int = 0

  override private[ml] def toOld(id: Int): OldNode = {
    new OldNode(id, new OldPredict(prediction, prob = impurityStats.prob(prediction)),
      impurity, isLeaf = true, None, None, None, None)
  }

  override private[ml] def maxSplitFeatureIndex(): Int = -1

  override private[tree] def deepCopy(): Node = {
    new LeafNodeV2(prediction, impurity, impurityStats, sortedLabels)
  }
}


