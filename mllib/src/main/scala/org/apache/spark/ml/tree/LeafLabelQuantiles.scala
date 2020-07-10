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
import math.floor

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator
import org.apache.spark.mllib.tree.model.{ImpurityStats, InformationGainStats => OldInformationGainStats, Node => OldNode, Predict => OldPredict}

class LeafLabelQuantiles(quantiles: Option[ ArrayBuffer[Float]]) extends Serializable {
  def expose() : ArrayBuffer[Float] = {
    quantiles match {
      case Some(q) => q
      case None => ArrayBuffer[Float]()
    }
  }
}

object LeafLabelQuantiles {
  def make(sortedLabels : ArrayBuffer[Float]) : LeafLabelQuantiles = {
    if (sortedLabels.size < 2) {
      new LeafLabelQuantiles(None)
    } else {
      val x = sortedLabels
      val quantiles = new ArrayBuffer[Float](101)
      quantiles += x(0)
      for (i <- 1 to 99) {
        val p = i.toDouble / 100.0
        val h = (x.size - 1).toDouble * p
        val floor_h = floor(h)
        quantiles += (x(floor_h.toInt).toDouble + (h - floor_h) * (x(floor_h.toInt + 1).toDouble - x(floor_h.toInt).toDouble)).toFloat
      }
      quantiles += x(x.size - 1)
      new LeafLabelQuantiles(Some(quantiles))
    }
  }
  def size : Int = { 101 }
}
