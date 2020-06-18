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

"""
Random Forest Regressor Example.
"""
from __future__ import print_function

# $example on$
from pyspark.ml import Pipeline
#from pyspark.ml.regression import RandomForestRegressor
from quantile_random_forests import QuantileRandomForestRegressor, decisionTreeToDebugStringV2
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
# $example off$
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestRegressorExample")\
        .getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = QuantileRandomForestRegressor(featuresCol="indexedFeatures")

    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, rf])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Make predictions.
    model.stages[1].set_transform_mode("distribution", bins = 10)
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select(expr("prediction._1 as predicted_avg"), expr("prediction._2 as upper_bounds"), expr("prediction._3 as lower_bounds"), expr("prediction._4 as densities"), "label", "features").show(5, truncate = False)

    rfModel = model.stages[1]
    print(rfModel)  # summary only
    # $example off$

    print(decisionTreeToDebugStringV2(spark, rfModel.trees[0]))

    spark.stop()
