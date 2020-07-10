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

  test("leaf label quantiles") {
    val expected = ArrayBuffer(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0)
    val ans = LeafLabelQuantiles.make(ArrayBuffer(0.0f, 50.0f, 100.0f)).expose
    assert(expected == ans)
    assert(101 == ans.size)

    val b = ArrayBuffer[Float]()
    for (i <- 0 to 1000) {
      b += i.toFloat
    }
    val bexpected = ArrayBuffer(0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0, 510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0, 610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0, 710.0, 720.0, 730.0, 740.0, 750.0, 760.0, 770.0, 780.0, 790.0, 800.0, 810.0, 820.0, 830.0, 840.0, 850.0, 860.0, 870.0, 880.0, 890.0, 900.0, 910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0, 1000.0)
    assert(bexpected == LeafLabelQuantiles.make(b).expose)
  }

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
    assert(LeafLabelQuantiles.make(ArrayBuffer[Float](1.0f,1.0f,1.0f,1.0f,1.0f)).expose == tree.rootNode.getLabels)
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
    assert(LeafLabelQuantiles.make(ArrayBuffer[Float](1.0f,1.0f,1.0f,1.0f,1.0f)).expose == tree2.rootNode.getLabels)
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
    assert(LeafLabelQuantiles.make(ArrayBuffer(0.0f,0.0f,0.0f,0.0f)).expose == model.rootNode.predictImpl(Vectors.dense(1.0)).getLabels)
    assert(LeafLabelQuantiles.make(ArrayBuffer(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f)).expose == model.rootNode.predictImpl(Vectors.dense(2.0)).getLabels)
    assert(LeafLabelQuantiles.make(ArrayBuffer(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f)).expose == model.rootNode.predictImpl(Vectors.dense(0.0)).getLabels)
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
    assert(LeafLabelQuantiles.make(ArrayBuffer(0.0f)).expose == prunedTree.rootNode.predictImpl(Vectors.dense(0.0,0.0)).getLabels)
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
    assert(LeafLabelQuantiles.make(ArrayBuffer(0.0f,0.0f)).expose == prunedTree.rootNode.predictImpl(Vectors.dense(0.0,0.0)).getLabels)
    assert(prunedTree.rootNode.predictImpl(Vectors.dense(0.0,0.0)).prediction == 0.0)
    assert(LeafLabelQuantiles.make(ArrayBuffer(0.0f,0.5f,1.0f)).expose == prunedTree.rootNode.predictImpl(Vectors.dense(1.0,1.0)).getLabels)
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
