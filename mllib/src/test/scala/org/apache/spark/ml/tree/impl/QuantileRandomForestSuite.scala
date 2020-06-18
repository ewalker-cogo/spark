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

package org.apache.spark.ml.tree.impl

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.tree._
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.tree.{DecisionTreeSuite => OldDTSuite, EnsembleTestHelper}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, QuantileStrategy, Strategy => OldStrategy}
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, GiniCalculator, Variance}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.util.collection.OpenHashMap

/**
 * Test suite for [[RandomForest]].
 */
class QuantileRandomForestSuite extends SparkFunSuite with MLlibTestSparkContext {

  import QuantileRandomForestSuite.mapToVec

  private val seed = 42

  test("train with empty arrays") {
    val lp = LabeledPoint(1.0, Vectors.dense(Array.empty[Double]))
    val data = Array.fill(5)(lp)
    val rdd = sc.parallelize(data)

    val strategy = new OldStrategy(OldAlgo.Regression, Gini, maxDepth = 2,
      maxBins = 5)
    withClue("DecisionTree requires number of features > 0," +
      " but was given an empty features vector") {
      intercept[IllegalArgumentException] {
        QuantileRandomForest.run(rdd, strategy, 1, "all", 42L, instr = None)
      }
    }
  }

  test("train with constant features") {
    val lp = LabeledPoint(1.0, Vectors.dense(0.0, 0.0, 0.0))
    val data = Array.fill(5)(lp)
    val rdd = sc.parallelize(data)
    val strategy = new OldStrategy(
          OldAlgo.Classification,
          Gini,
          maxDepth = 2,
          numClasses = 2,
          maxBins = 5,
          categoricalFeaturesInfo = Map(0 -> 1, 1 -> 5))
    val Array(tree) = QuantileRandomForest.run(rdd, strategy, 1, "all", 42L, instr = None)
    assert(tree.rootNode.impurity === -1.0)
    assert(tree.depth === 0)
    assert(tree.rootNode.prediction === lp.label)
    import QuantileRandomForestImplicits._
    assert(ArrayBuffer[Float](1.0f,1.0f,1.0f,1.0f,1.0f) == tree.rootNode.getLabels)
    import DecisionTreeModelImplicits._
    assert(tree.toDebugStringV2.startsWith("DecisionTreeClassificationModel"))

    // Test with no categorical features
    val strategy2 = new OldStrategy(
      OldAlgo.Regression,
      Variance,
      maxDepth = 2,
      maxBins = 5)
    val Array(tree2) = QuantileRandomForest.run(rdd, strategy2, 1, "all", 42L, instr = None)
    assert(tree2.rootNode.impurity === -1.0)
    assert(tree2.depth === 0)
    assert(tree2.rootNode.prediction === lp.label)
    assert(ArrayBuffer[Float](1.0f,1.0f,1.0f,1.0f,1.0f) == tree2.rootNode.getLabels)
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of other algorithm internals
  /////////////////////////////////////////////////////////////////////////////

  test("Use soft prediction for binary classification with ordered categorical features") {
    // The following dataset is set up such that the best split is {1} vs. {0, 2}.
    // If the hard prediction is used to order the categories, then {0} vs. {1, 2} is chosen.
    val arr = Array(
      LabeledPoint(0.0, Vectors.dense(0.0)),
      LabeledPoint(0.0, Vectors.dense(0.0)),
      LabeledPoint(0.0, Vectors.dense(0.0)),
      LabeledPoint(1.0, Vectors.dense(0.0)),
      LabeledPoint(0.0, Vectors.dense(1.0)),
      LabeledPoint(0.0, Vectors.dense(1.0)),
      LabeledPoint(0.0, Vectors.dense(1.0)),
      LabeledPoint(0.0, Vectors.dense(1.0)),
      LabeledPoint(0.0, Vectors.dense(2.0)),
      LabeledPoint(0.0, Vectors.dense(2.0)),
      LabeledPoint(0.0, Vectors.dense(2.0)),
      LabeledPoint(1.0, Vectors.dense(2.0)))
    val input = sc.parallelize(arr)

    // Must set maxBins s.t. the feature will be treated as an ordered categorical feature.
    val strategy = new OldStrategy(algo = OldAlgo.Classification, impurity = Gini, maxDepth = 1,
      numClasses = 2, categoricalFeaturesInfo = Map(0 -> 3), maxBins = 3)

    val model = QuantileRandomForest.run(input, strategy, numTrees = 1, featureSubsetStrategy = "all",
      seed = 42, instr = None, prune = false).head

    model.rootNode match {
      case n: InternalNode => n.split match {
        case s: CategoricalSplit =>
          assert(s.leftCategories === Array(1.0))
        case _ => throw new AssertionError("model.rootNode.split was not a CategoricalSplit")
      }
      case _ => throw new AssertionError("model.rootNode was not an InternalNode")
    }
    import QuantileRandomForestImplicits._
    assert(ArrayBuffer(0.0f,0.0f,0.0f,0.0f) == model.rootNode.predictImpl(Vectors.dense(1.0)).getLabels)
    assert(ArrayBuffer(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f) == model.rootNode.predictImpl(Vectors.dense(2.0)).getLabels)
    assert(ArrayBuffer(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f) == model.rootNode.predictImpl(Vectors.dense(0.0)).getLabels)
  }

  test("Second level node building with vs. without groups") {
    val arr = OldDTSuite.generateOrderedLabeledPoints().map(_.asML)
    assert(arr.length === 1000)
    val rdd = sc.parallelize(arr)
    // For tree with 1 group
    val strategy1 =
      new OldStrategy(OldAlgo.Classification, Entropy, 3, 2, 100, maxMemoryInMB = 1000)
    // For tree with multiple groups
    val strategy2 =
      new OldStrategy(OldAlgo.Classification, Entropy, 3, 2, 100, maxMemoryInMB = 0)

    val tree1 = RandomForest.run(rdd, strategy1, numTrees = 1, featureSubsetStrategy = "all",
      seed = 42, instr = None).head
    val tree2 = RandomForest.run(rdd, strategy2, numTrees = 1, featureSubsetStrategy = "all",
      seed = 42, instr = None).head

    def getChildren(rootNode: Node): Array[InternalNode] = rootNode match {
      case n: InternalNode =>
        assert(n.leftChild.isInstanceOf[InternalNode])
        assert(n.rightChild.isInstanceOf[InternalNode])
        Array(n.leftChild.asInstanceOf[InternalNode], n.rightChild.asInstanceOf[InternalNode])
      case _ => throw new AssertionError("rootNode was not an InternalNode")
    }

    // Single group second level tree construction.
    val children1 = getChildren(tree1.rootNode)
    val children2 = getChildren(tree2.rootNode)

    // Verify whether the splits obtained using single group and multiple group level
    // construction strategies are the same.
    for (i <- 0 until 2) {
      assert(children1(i).gain > 0)
      assert(children2(i).gain > 0)
      assert(children1(i).split === children2(i).split)
      assert(children1(i).impurity === children2(i).impurity)
      assert(children1(i).impurityStats.stats === children2(i).impurityStats.stats)
      assert(children1(i).leftChild.impurity === children2(i).leftChild.impurity)
      assert(children1(i).rightChild.impurity === children2(i).rightChild.impurity)
      assert(children1(i).prediction === children2(i).prediction)
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Tests for pruning of redundant subtrees (generated by a split improving the
  // impurity measure, but always leading to the same prediction).
  ///////////////////////////////////////////////////////////////////////////////

  test("SPARK-3159 tree model redundancy - classification") {
    // The following dataset is set up such that splitting over feature_1 for points having
    // feature_0 = 0 improves the impurity measure, despite the prediction will always be 0
    // in both branches.
    val arr = Array(
      LabeledPoint(0.0, Vectors.dense(0.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(0.0, 1.0)),
      LabeledPoint(0.0, Vectors.dense(0.0, 0.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0)),
      LabeledPoint(0.0, Vectors.dense(1.0, 0.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0))
    )
    val rdd = sc.parallelize(arr)

    val numClasses = 2
    val strategy = new OldStrategy(algo = OldAlgo.Classification, impurity = Gini, maxDepth = 4,
      numClasses = numClasses, maxBins = 32)

    val prunedTree = QuantileRandomForest.run(rdd, strategy, numTrees = 1, featureSubsetStrategy = "auto",
      seed = 42, instr = None).head

    val unprunedTree = QuantileRandomForest.run(rdd, strategy, numTrees = 1, featureSubsetStrategy = "auto",
      seed = 42, instr = None, prune = false).head

    //assert(prunedTree.numNodes === 5)
    assert(prunedTree.numNodes === 7)
    assert(unprunedTree.numNodes === 7)

    assert(QuantileRandomForestSuite.getSumLeafCounters(List(prunedTree.rootNode)) === arr.size)
    import QuantileRandomForestImplicits._
    assert(ArrayBuffer(0.0f) == prunedTree.rootNode.predictImpl(Vectors.dense(0.0,0.0)).getLabels)
    assert(prunedTree.rootNode.predictImpl(Vectors.dense(0.0,0.0)).prediction == 0.0)
  }

  test("SPARK-3159 tree model redundancy - regression") {
    // The following dataset is set up such that splitting over feature_0 for points having
    // feature_1 = 1 improves the impurity measure, despite the prediction will always be 0.5
    // in both branches.
    val arr = Array(
      LabeledPoint(0.0, Vectors.dense(0.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(0.0, 1.0)),
      LabeledPoint(0.0, Vectors.dense(0.0, 0.0)),
      LabeledPoint(0.0, Vectors.dense(1.0, 0.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(0.0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(0.5, Vectors.dense(1.0, 1.0))
    )
    val rdd = sc.parallelize(arr)

    val strategy = new OldStrategy(algo = OldAlgo.Regression, impurity = Variance, maxDepth = 4,
      numClasses = 0, maxBins = 32)

    val prunedTree = QuantileRandomForest.run(rdd, strategy, numTrees = 1, featureSubsetStrategy = "auto",
      seed = 42, instr = None).head

    val unprunedTree = QuantileRandomForest.run(rdd, strategy, numTrees = 1, featureSubsetStrategy = "auto",
      seed = 42, instr = None, prune = false).head

    //assert(prunedTree.numNodes === 3)
    assert(prunedTree.numNodes === 5)
    assert(unprunedTree.numNodes === 5)
    assert(QuantileRandomForestSuite.getSumLeafCounters(List(prunedTree.rootNode)) === arr.size)
    import QuantileRandomForestImplicits._
    assert(ArrayBuffer(0.0f,0.0f) == prunedTree.rootNode.predictImpl(Vectors.dense(0.0,0.0)).getLabels)
    assert(prunedTree.rootNode.predictImpl(Vectors.dense(0.0,0.0)).prediction == 0.0)
    assert(ArrayBuffer(0.0f,0.5f,1.0f) == prunedTree.rootNode.predictImpl(Vectors.dense(1.0,1.0)).getLabels)
    assert(prunedTree.rootNode.predictImpl(Vectors.dense(1.0,1.0)).prediction == 0.5)
  }
}

private object QuantileRandomForestSuite {
  def mapToVec(map: Map[Int, Double]): Vector = {
    val size = (map.keys.toSeq :+ 0).max + 1
    val (indices, values) = map.toSeq.sortBy(_._1).unzip
    Vectors.sparse(size, indices.toArray, values.toArray)
  }

  @tailrec
  private def getSumLeafCounters(nodes: List[Node], acc: Long = 0): Long = {
    if (nodes.isEmpty) {
      acc
    }
    else {
      nodes.head match {
        case i: InternalNode => getSumLeafCounters(i.leftChild :: i.rightChild :: nodes.tail, acc)
        case l: LeafNode => getSumLeafCounters(nodes.tail, acc + l.impurityStats.count)
      }
    }
  }
}
