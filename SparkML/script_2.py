from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, NaiveBayes, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, Imputer, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession, DataFrame
import os

PATH = './SparkML/mushrooms.csv'
SEED = 12345


def set_up_env():
    os.environ["SPARK_HOME"] = "/home/timurezer/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/timurezer/anaconda3/envs/mlbd/bin/python"


def getSparkSession():
    set_up_env()
    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("Mashrooms") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def readDataset(spark, show=False):
    mushrooms = spark.read \
        .option("delimiter", ",") \
        .option("inferSchema", "true") \
        .option("header", "true") \
        .csv(PATH)

    if show:
        mushrooms.printSchema()
        mushrooms.show()
    return mushrooms


class MyImputer(Transformer):
    def __init__(self, column_names):
        super(MyImputer, self).__init__()
        self.column_names = column_names
        self.imputer = Imputer(strategy="mode",
                               inputCols=column_names,
                               outputCols=[x + "_" for x in column_names])

    def _transform(self, df):
        df = self.imputer.fit(df).transform(df)
        for column in self.column_names:
            df = df.drop(column).withColumnRenamed(column + "_", column)
        return df


class ColumnsIndexer(Transformer):
    def __init__(self, column_names):
        super(ColumnsIndexer, self).__init__()
        self.column_names = column_names

    def _transform(self, df: DataFrame) -> DataFrame:
        for column in self.column_names:
            columnIndexer = StringIndexer(inputCol=column,
                                          outputCol=column + "_",
                                          handleInvalid="keep")
            df = columnIndexer.fit(df).transform(df).drop(column) \
                .withColumnRenamed(column + "_", column)
        return df


class MyOneHotEncoder(Transformer):
    def __init__(self, column_names):
        super(MyOneHotEncoder, self).__init__()
        self.column_names = column_names

    def _transform(self, df: DataFrame) -> DataFrame:
        for column in self.column_names:
            encoder = OneHotEncoder(inputCol=column,
                                    outputCol=column + "_")
            df = encoder.fit(df).transform(df).drop(column) \
                .withColumnRenamed(column + "_", column)
        return df


def run_acount_acc(pipeline, X_train, X_test):
    model = pipeline.fit(X_train)
    rawPredictions = model.transform(X_test)
    evaluator = MulticlassClassificationEvaluator(labelCol="class",
                                                  predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(rawPredictions)
    return accuracy


def main():
    spark = getSparkSession()

    mushrooms = readDataset(spark)
    columns = mushrooms.columns
    columns_no_target = [x for x in mushrooms.columns if x != "class"]

    indexer = ColumnsIndexer(columns)
    mushrooms = indexer.transform(mushrooms)

    X_train, X_test = mushrooms.randomSplit([0.7, 0.3], seed=SEED)
    X_train.cache()
    X_test.cache()

    imputer = MyImputer(columns)
    encoder = MyOneHotEncoder(columns_no_target)
    assembler = VectorAssembler(inputCols=columns_no_target, outputCol="features")

    trainer_reg = LogisticRegression(labelCol="class", featuresCol="features")
    trainer_bayes = NaiveBayes(labelCol="class", featuresCol="features")
    trainer_svm = LinearSVC(labelCol="class", featuresCol="features")
    trainer_forest = RandomForestClassifier(labelCol="class", featuresCol="features")

    pipeline_reg = Pipeline(stages=[imputer, encoder, assembler, trainer_reg])
    pipeline_bayes = Pipeline(stages=[imputer, encoder, assembler, trainer_bayes])
    pipeline_svm = Pipeline(stages=[imputer, encoder, assembler, trainer_svm])
    pipeline_forest = Pipeline(stages=[imputer, encoder, assembler, trainer_forest])

    print("Let's use OneHotEncoding!")

    scores = [[run_acount_acc(pipeline_reg, X_train, X_test), "pipeline_regression"],
              [run_acount_acc(pipeline_bayes, X_train, X_test), "pipeline_bayes"],
              [run_acount_acc(pipeline_svm, X_train, X_test), "pipeline_svm"],
              [run_acount_acc(pipeline_forest, X_train, X_test), "pipeline_forest"]]

    print('accuracies:')
    for x, name in scores:
        print(f'{name}: {x}')

    spark.stop()


if __name__ == '__main__':
    main()
