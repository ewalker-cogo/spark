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

object DecisionTreeModelImplicits {
  implicit class TreeWithMoreDebug(n: DecisionTreeModel) {
    /** Full description of model */
    def toDebugStringV2: String = {
      import QuantileRandomForestImplicits._
      val header = n.toString + "\n"
      header + n.rootNode.subtreeToStringV2(2)
    }
  }
}

