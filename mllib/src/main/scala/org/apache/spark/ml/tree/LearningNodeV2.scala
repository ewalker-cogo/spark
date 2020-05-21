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
 * Version of a node used in learning.  This uses vars so that we can modify nodes as we split the
 * tree by adding children, etc.
 *
 * For now, we use node IDs.  These will be kept internal since we hope to remove node IDs
 * in the future, or at least change the indexing (so that we can support much deeper trees).
 *
 * This node can either be:
 *  - a leaf node, with leftChild, rightChild, split set to null, or
 *  - an internal node, with all values set
 *
 * @param id  We currently use the same indexing as the old implementation in
 *            [[org.apache.spark.mllib.tree.model.Node]], but this will change later.
 * @param isLeaf  Indicates whether this node will definitely be a leaf in the learned tree,
 *                so that we do not need to consider splitting it further.
 * @param stats  Impurity statistics for this node.
 */
private[tree] class LearningNodeV2(
    var id: Int,
    var leftChild: Option[LearningNodeV2],
    var rightChild: Option[LearningNodeV2],
    var split: Option[Split],
    var isLeaf: Boolean,
    var stats: ImpurityStats,
    var labels: Option[ArrayBuffer[Float]] = None) extends Serializable {

  def toNode: Node = toNode(prune = true)

  /**
   * Convert this [[LearningNodeV2]] to a regular [[Node]], and recurse on any children.
   */
  def toNode(prune: Boolean = true): Node = {

    if (!leftChild.isEmpty || !rightChild.isEmpty) {
      assert(leftChild.nonEmpty && rightChild.nonEmpty && split.nonEmpty && stats != null,
        "Unknown error during Decision Tree learning.  Could not convert LearningNodeV2 to Node.")
      (leftChild.get.toNode(prune), rightChild.get.toNode(prune)) match {
        case (l: LeafNodeV2, r: LeafNodeV2) if prune && l.prediction == r.prediction =>
          new LeafNodeV2(l.prediction, stats.impurity, stats.impurityCalculator, Some(l.getLabels ++ r.getLabels))
        case (l, r) =>
          new InternalNode(stats.impurityCalculator.predict, stats.impurity, stats.gain,
            l, r, split.get, stats.impurityCalculator)
      }
    } else {
      if (stats.valid) {
        new LeafNodeV2(stats.impurityCalculator.predict, stats.impurity,
          stats.impurityCalculator, labels)
      } else {
        // Here we want to keep same behavior with the old mllib.DecisionTreeModel
        new LeafNodeV2(stats.impurityCalculator.predict, -1.0, stats.impurityCalculator, labels)
      }
    }
  }

  /**
   * Get the node index corresponding to this data point.
   * This function mimics prediction, passing an example from the root node down to a leaf
   * or unsplit node; that node's index is returned.
   *
   * @param binnedFeatures  Binned feature vector for data point.
   * @param splits possible splits for all features, indexed (numFeatures)(numSplits)
   * @return Leaf index if the data point reaches a leaf.
   *         Otherwise, last node reachable in tree matching this example.
   *         Note: This is the global node index, i.e., the index used in the tree.
   *         This index is different from the index used during training a particular
   *         group of nodes on one call to
   *         [[org.apache.spark.ml.tree.impl.RandomForest.findBestSplits()]].
   */
  def predictImpl(binnedFeatures: Array[Int], splits: Array[Array[Split]]): Int = {
    if (this.isLeaf || this.split.isEmpty) {
      this.id
    } else {
      val split = this.split.get
      val featureIndex = split.featureIndex
      val splitLeft = split.shouldGoLeft(binnedFeatures(featureIndex), splits(featureIndex))
      if (this.leftChild.isEmpty) {
        // Not yet split. Return next layer of nodes to train
        if (splitLeft) {
          LearningNodeV2.leftChildIndex(this.id)
        } else {
          LearningNodeV2.rightChildIndex(this.id)
        }
      } else {
        if (splitLeft) {
          this.leftChild.get.predictImpl(binnedFeatures, splits)
        } else {
          this.rightChild.get.predictImpl(binnedFeatures, splits)
        }
      }
    }
  }


}

private[tree] object LearningNodeV2 {

  /** Create a node with some of its fields set. */
  def apply(
      id: Int,
      isLeaf: Boolean,
      stats: ImpurityStats): LearningNodeV2 = {
    new LearningNodeV2(id, None, None, None, false, stats)
  }

  /** Create an empty node with the given node index.  Values must be set later on. */
  def emptyNode(nodeIndex: Int): LearningNodeV2 = {
    new LearningNodeV2(nodeIndex, None, None, None, false, null)
  }

  // The below indexing methods were copied from spark.mllib.tree.model.Node

  /**
   * Return the index of the left child of this node.
   */
  def leftChildIndex(nodeIndex: Int): Int = nodeIndex << 1

  /**
   * Return the index of the right child of this node.
   */
  def rightChildIndex(nodeIndex: Int): Int = (nodeIndex << 1) + 1

  /**
   * Get the parent index of the given node, or 0 if it is the root.
   */
  def parentIndex(nodeIndex: Int): Int = nodeIndex >> 1

  /**
   * Return the level of a tree which the given node is in.
   */
  def indexToLevel(nodeIndex: Int): Int = if (nodeIndex == 0) {
    throw new IllegalArgumentException(s"0 is not a valid node index.")
  } else {
    java.lang.Integer.numberOfTrailingZeros(java.lang.Integer.highestOneBit(nodeIndex))
  }

  /**
   * Returns true if this is a left child.
   * Note: Returns false for the root.
   */
  def isLeftChild(nodeIndex: Int): Boolean = nodeIndex > 1 && nodeIndex % 2 == 0

  /**
   * Return the maximum number of nodes which can be in the given level of the tree.
   * @param level  Level of tree (0 = root).
   */
  def maxNodesInLevel(level: Int): Int = 1 << level

  /**
   * Return the index of the first node in the given level.
   * @param level  Level of tree (0 = root).
   */
  def startIndexInLevel(level: Int): Int = 1 << level

  /**
   * Traces down from a root node to get the node with the given node index.
   * This assumes the node exists.
   */
  def getNode(nodeIndex: Int, rootNode: LearningNodeV2): LearningNodeV2 = {
    var tmpNode: LearningNodeV2 = rootNode
    var levelsToGo = indexToLevel(nodeIndex)
    while (levelsToGo > 0) {
      if ((nodeIndex & (1 << levelsToGo - 1)) == 0) {
        tmpNode = tmpNode.leftChild.get
      } else {
        tmpNode = tmpNode.rightChild.get
      }
      levelsToGo -= 1
    }
    tmpNode
  }

}
