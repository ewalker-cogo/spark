#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import warnings

from pyspark import since, keyword_only
from pyspark.ml.param.shared import *
from pyspark.ml.util import *
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaWrapper
from pyspark.ml.common import inherit_doc
from pyspark.sql import DataFrame

from pyspark.ml.regression import TreeRegressorParams

from pyspark.ml.regression import RandomForestParams

@inherit_doc
class QuantileRandomForestRegressor(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasSeed,
                                    RandomForestParams, TreeRegressorParams, HasCheckpointInterval,
                                    JavaMLWritable, JavaMLReadable):
    """
    `Random Forest <http://en.wikipedia.org/wiki/Random_forest>`_
    learning algorithm for regression.
    It supports both continuous and categorical features.

    >>> from numpy import allclose
    >>> from pyspark.ml.linalg import Vectors
    >>> df = spark.createDataFrame([
    ...     (1.0, Vectors.dense(1.0)),
    ...     (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
    >>> rf = RandomForestRegressor(numTrees=2, maxDepth=2, seed=42)
    >>> model = rf.fit(df)
    >>> model.featureImportances
    SparseVector(1, {0: 1.0})
    >>> allclose(model.treeWeights, [1.0, 1.0])
    True
    >>> test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
    >>> model.transform(test0).head().prediction
    0.0
    >>> model.numFeatures
    1
    >>> model.trees
    [DecisionTreeRegressionModel (uid=...) of depth..., DecisionTreeRegressionModel...]
    >>> model.getNumTrees
    2
    >>> test1 = spark.createDataFrame([(Vectors.sparse(1, [0], [1.0]),)], ["features"])
    >>> model.transform(test1).head().prediction
    0.5
    >>> rfr_path = temp_path + "/rfr"
    >>> rf.save(rfr_path)
    >>> rf2 = RandomForestRegressor.load(rfr_path)
    >>> rf2.getNumTrees()
    2
    >>> model_path = temp_path + "/rfr_model"
    >>> model.save(model_path)
    >>> model2 = RandomForestRegressionModel.load(model_path)
    >>> model.featureImportances == model2.featureImportances
    True

    .. versionadded:: 1.4.0
    """

    @keyword_only
    def __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                 maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                 maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10,
                 impurity="variance", subsamplingRate=1.0, seed=None, numTrees=20,
                 featureSubsetStrategy="auto"):
        """
        __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction", \
                 maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, \
                 maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, \
                 impurity="variance", subsamplingRate=1.0, seed=None, numTrees=20, \
                 featureSubsetStrategy="auto")
        """
        super(QuantileRandomForestRegressor, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.regression.QuantileRandomForestRegressor", self.uid)
        self._setDefault(maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                         maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10,
                         impurity="variance", subsamplingRate=1.0, numTrees=20,
                         featureSubsetStrategy="auto")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("1.4.0")
    def setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                  maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                  maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10,
                  impurity="variance", subsamplingRate=1.0, seed=None, numTrees=20,
                  featureSubsetStrategy="auto"):
        """
        setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction", \
                  maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, \
                  maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, \
                  impurity="variance", subsamplingRate=1.0, seed=None, numTrees=20, \
                  featureSubsetStrategy="auto")
        Sets params for linear regression.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return QuantileRandomForestRegressionModel(java_model)

    @since("2.4.0")
    def setFeatureSubsetStrategy(self, value):
        """
        Sets the value of :py:attr:`featureSubsetStrategy`.
        """
        return self._set(featureSubsetStrategy=value)


from pyspark.ml.regression import TreeEnsembleModel

class QuantileRandomForestRegressionModel(TreeEnsembleModel, JavaPredictionModel, JavaMLWritable,
                                          JavaMLReadable):
    """
    Model fitted by :class:`QuantileRandomForestRegressor`.

    .. versionadded:: 1.4.0
    """

    @property
    @since("2.0.0")
    def trees(self):
        """Trees in this ensemble. Warning: These have null parent Estimators."""
        return [DecisionTreeRegressionModel(m) for m in list(self._call_java("trees"))]

    @property
    @since("2.0.0")
    def featureImportances(self):
        """
        Estimate of the importance of each feature.

        Each feature's importance is the average of its importance across all trees in the ensemble
        The importance vector is normalized to sum to 1. This method is suggested by Hastie et al.
        (Hastie, Tibshirani, Friedman. "The Elements of Statistical Learning, 2nd Edition." 2001.)
        and follows the implementation from scikit-learn.

        .. seealso:: :py:attr:`DecisionTreeRegressionModel.featureImportances`
        """
        return self._call_java("featureImportances")


    def set_transform_mode(self, mode):
        if mode not in ("observations", "predictions"):
            raise Exception("bad transform mode: " + mode)
        return self._call_java("setTransformMode", mode)
