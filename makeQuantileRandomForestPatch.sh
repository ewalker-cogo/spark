P=`pwd`
mkdir -p QUANTILE_RANDOM_FOREST_PATCH
cd mllib/target/scala-2.11/classes
jar cvf $P/QUANTILE_RANDOM_FOREST_PATCH/quantileRandomForest.jar \
org/apache/spark/ml/regression/QuantileRandomForest*.class \
org/apache/spark/ml/regression/TreeLabelWalker*.class \
org/apache/spark/ml/regression/ForestWalker*.class \
org/apache/spark/ml/tree/LearningNodeV2*.class \
org/apache/spark/ml/tree/LeafNodeV2*.class \
org/apache/spark/ml/tree/QuantileRandomForestImplicits*.class \
org/apache/spark/ml/tree/impl/QuantileRandomForest*.class
Q=$P/QUANTILE_RANDOM_FOREST_PATCH/
cd $P
cp python/pyspark/ml/quantile_random_forests.py $Q
cp examples/src/main/python/ml/quantile_random*.py $Q
cp data/mllib/sample_libsvm_data.txt $Q

