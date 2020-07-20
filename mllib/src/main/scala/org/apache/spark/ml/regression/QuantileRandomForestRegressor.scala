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
import math.floor

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

/**
 * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a>
 * learning algorithm for regression.
 * It supports both continuous and categorical features.
 */
class QuantileRandomForestRegressor(override val uid: String)
  extends Predictor[Vector, QuantileRandomForestRegressor, QuantileRandomForestRegressionModel]
  with RandomForestRegressorParams with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("rfr"))

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeRegressorParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
   * Specifies how often to checkpoint the cached node IDs.
   * E.g. 10 means that the cache will get checkpointed every 10 iterations.
   * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
   * [[org.apache.spark.SparkContext]].
   * Must be at least 1.
   * (default = 10)
   * @group setParam
   */
  @Since("1.4.0")
  override def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setImpurity(value: String): this.type = set(impurity, value)

  // Parameters from TreeEnsembleParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setSeed(value: Long): this.type = set(seed, value)

  // Parameters from RandomForestParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setNumTrees(value: Int): this.type = set(numTrees, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setFeatureSubsetStrategy(value: String): this.type =
    set(featureSubsetStrategy, value)

  override protected def train(
      dataset: Dataset[_]): QuantileRandomForestRegressionModel = instrumented { instr =>
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity)

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, labelCol, featuresCol, predictionCol, impurity, numTrees,
      featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, seed, subsamplingRate, cacheNodeIds, checkpointInterval)

    val trees = QuantileRandomForest
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeRegressionModel])

    val numFeatures = oldDataset.first().features.size
    instr.logNamedValue(Instrumentation.loggerTags.numFeatures, numFeatures)
    new QuantileRandomForestRegressionModel(uid, trees, numFeatures)
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): QuantileRandomForestRegressor = defaultCopy(extra)
}

@Since("1.4.0")
object QuantileRandomForestRegressor extends DefaultParamsReadable[QuantileRandomForestRegressor]{
  /** Accessor for supported impurity settings: variance */
  @Since("1.4.0")
  final val supportedImpurities: Array[String] = TreeRegressorParams.supportedImpurities

  /** Accessor for supported featureSubsetStrategy settings: auto, all, onethird, sqrt, log2 */
  @Since("1.4.0")
  final val supportedFeatureSubsetStrategies: Array[String] =
    TreeEnsembleParams.supportedFeatureSubsetStrategies

  @Since("2.0.0")
  override def load(path: String): QuantileRandomForestRegressor = super.load(path)

}

/**
 * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a> model for regression.
 * It supports both continuous and categorical features.
 *
 * @param _trees  Decision trees in the ensemble.
 * @param numFeatures  Number of features used by this model
 */
@Since("1.4.0")
class QuantileRandomForestRegressionModel private[ml] (
    override val uid: String,
    private val _trees: Array[DecisionTreeRegressionModel],
    override val numFeatures: Int)
  extends PredictionModel[Vector, QuantileRandomForestRegressionModel]
  with RandomForestRegressorParams with TreeEnsembleModel[DecisionTreeRegressionModel]
  with MLWritable with Serializable {

  require(_trees.nonEmpty, "QuantileRandomForestRegressionModel requires at least 1 tree.")

  /**
   * Construct a random forest regression model, with all trees weighted equally.
   *
   * @param trees  Component trees
   */
  private[ml] def this(trees: Array[DecisionTreeRegressionModel], numFeatures: Int) =
    this(Identifiable.randomUID("qrfr"), trees, numFeatures)

  @Since("1.4.0")
  override def trees: Array[DecisionTreeRegressionModel] = _trees

  // Note: We may add support for weights (based on tree performance) later on.
  private lazy val _treeWeights: Array[Double] = Array.fill[Double](_trees.length)(1.0)

  @Since("1.4.0")
  override def treeWeights: Array[Double] = _treeWeights

  def setTransformMode(mode:String, bins: Int = 0) : Unit = {
    throw new Exception("not supported with average quantiles implementation")
  }

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val distUDF = udf { (features: Any) =>
      bcastModel.value.distribution(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), distUDF(col($(featuresCol))))
  }

  override def predict(features: Vector): Double = {
    // TODO: When we add a generic Bagging class, handle transform there.  SPARK-7128
    // Predict average of tree predictions.
    // Ignore the weights since all are 1.0 for now.
    _trees.map(_.rootNode.predictImpl(features).prediction).sum / getNumTrees
  }

  def distribution(features: Vector) : (Double, ArrayBuffer[Double]) = {
    import QuantileRandomForestImplicits._
    var predictionSum : Double = 0.0
    val numTrees = _trees.size
    val quantilesFromTrees = new ArrayBuffer[Float](numTrees * LeafLabelQuantiles.size)
    var minimum = Float.MaxValue
    var maximum = Float.MinValue

    for (tree <- _trees) {
      val leafNode = tree.rootNode.predictImpl(features)
      predictionSum += leafNode.prediction
      val leafQuantiles = leafNode.getLabels
      if (leafQuantiles.length == 101) {
        if (leafQuantiles(0) < minimum) {
          minimum = leafQuantiles(0)
        }
        if (leafQuantiles(100) > maximum) {
          maximum = leafQuantiles(100)
        }
        for (q <- 1 to 99) {
          quantilesFromTrees += leafQuantiles(q)
        }
      }
    }
    val fromTreesSorted = quantilesFromTrees.sortWith(_ < _)
    val quantiles = new ArrayBuffer[Double](LeafLabelQuantiles.size)

    val x = fromTreesSorted
    quantiles += minimum.toDouble
    for (i <- 1 to 99) {
      val p = i.toDouble / 100.0
      val h = (x.size - 1).toDouble * p
      val floor_h = floor(h)
      quantiles += (x(floor_h.toInt).toDouble + (h - floor_h) * (x(floor_h.toInt + 1).toDouble - x(floor_h.toInt).toDouble)).toFloat
      }
    quantiles += maximum.toDouble

    (predictionSum / numTrees.toDouble, quantiles)
  }

  def PREV_AVG_BASED_distribution(features: Vector) : (Double, ArrayBuffer[Double]) = {
    import QuantileRandomForestImplicits._
    var predictionSum : Double = 0.0
    val numTrees = _trees.size
    val quantiles = new ArrayBuffer[Double](LeafLabelQuantiles.size)
    for (i <- 0 until LeafLabelQuantiles.size) {
        quantiles += 0.0
    }
    var N = 0
    for (tree <- _trees) {
      val leafNode = tree.rootNode.predictImpl(features)
      predictionSum += leafNode.prediction
      val leafQuantiles = leafNode.getLabels
      if (leafQuantiles.size > 0) {
        N += 1
        for (j <- 0 until leafQuantiles.size) {
          quantiles(j) = quantiles(j) + (leafQuantiles(j).toDouble - quantiles(j)) / N.toDouble
        }
      }
    }
    (predictionSum / numTrees.toDouble, quantiles)
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): QuantileRandomForestRegressionModel = {
    copyValues(new QuantileRandomForestRegressionModel(uid, _trees, numFeatures), extra).setParent(parent)
  }

  @Since("1.4.0")
  override def toString: String = {
    s"QuantileRandomForestRegressionModel (uid=$uid) with $getNumTrees trees"
  }

  /**
   * Estimate of the importance of each feature.
   *
   * Each feature's importance is the average of its importance across all trees in the ensemble
   * The importance vector is normalized to sum to 1. This method is suggested by Hastie et al.
   * (Hastie, Tibshirani, Friedman. "The Elements of Statistical Learning, 2nd Edition." 2001.)
   * and follows the implementation from scikit-learn.
   *
   * @see `DecisionTreeRegressionModel.featureImportances`
   */
  @Since("1.5.0")
  lazy val featureImportances: Vector = TreeEnsembleModel.featureImportances(trees, numFeatures)

  /** (private[ml]) Convert to a model in the old API */
  private[ml] def toOld: OldRandomForestModel = {
    new OldRandomForestModel(OldAlgo.Regression, _trees.map(_.toOld))
  }

  @Since("2.0.0")
  override def write: MLWriter =
    new QuantileRandomForestRegressionModel.QuantileRandomForestRegressionModelWriter(this)
}

@Since("2.0.0")
object QuantileRandomForestRegressionModel extends MLReadable[QuantileRandomForestRegressionModel] {

  @Since("2.0.0")
  override def read: MLReader[QuantileRandomForestRegressionModel] = new QuantileRandomForestRegressionModelReader

  @Since("2.0.0")
  override def load(path: String): QuantileRandomForestRegressionModel = super.load(path)

  private[QuantileRandomForestRegressionModel]
  class QuantileRandomForestRegressionModelWriter(instance: QuantileRandomForestRegressionModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      val extraMetadata: JObject = Map(
        "numFeatures" -> instance.numFeatures,
        "numTrees" -> instance.getNumTrees)
      EnsembleModelReadWrite.saveImpl(instance, path, sparkSession, extraMetadata)
    }
  }

  private class QuantileRandomForestRegressionModelReader extends MLReader[QuantileRandomForestRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[QuantileRandomForestRegressionModel].getName
    private val treeClassName = classOf[DecisionTreeRegressionModel].getName

    override def load(path: String): QuantileRandomForestRegressionModel = {
      implicit val format = DefaultFormats
      val (metadata: Metadata, treesData: Array[(Metadata, Node)], treeWeights: Array[Double]) =
        EnsembleModelReadWrite.loadImpl(path, sparkSession, className, treeClassName)
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val numTrees = (metadata.metadata \ "numTrees").extract[Int]

      val trees: Array[DecisionTreeRegressionModel] = treesData.map { case (treeMetadata, root) =>
        val tree =
          new DecisionTreeRegressionModel(treeMetadata.uid, root, numFeatures)
        treeMetadata.getAndSetParams(tree)
        tree
      }
      require(numTrees == trees.length, s"RandomForestRegressionModel.load expected $numTrees" +
        s" trees based on metadata but found ${trees.length} trees.")

      val model = new QuantileRandomForestRegressionModel(metadata.uid, trees, numFeatures)
      metadata.getAndSetParams(model)
      model
    }
  }

  /** Convert a model from the old API */
  //private[ml] def fromOld(
  //    oldModel: OldRandomForestModel,
  //    parent: RandomForestRegressor,
  //    categoricalFeatures: Map[Int, Int],
  //    numFeatures: Int = -1): RandomForestRegressionModel = {
  //  require(oldModel.algo == OldAlgo.Regression, "Cannot convert RandomForestModel" +
  //    s" with algo=${oldModel.algo} (old API) to RandomForestRegressionModel (new API).")
  //  val newTrees = oldModel.trees.map { tree =>
      // parent for each tree is null since there is no good way to set this.
  //    DecisionTreeRegressionModel.fromOld(tree, null, categoricalFeatures)
  //  }
  //  val uid = if (parent != null) parent.uid else Identifiable.randomUID("rfr")
  //  new RandomForestRegressionModel(uid, newTrees, numFeatures)
  //}
}
