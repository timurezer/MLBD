import os
from pyspark.sql import SparkSession
from pyspark.mllib.stat import Statistics

def set_up_env():
    os.environ["SPARK_HOME"] = "/home/timurezer/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/timurezer/anaconda3/envs/mlbd/bin/python"

def findNotOutliers(rdd):
    """
    count mean, std of array and fing items out of 3*sigma
    """
    # also it could be done with rdd.mean() and rdd.variance() :)
    parameters = rdd \
        .map(lambda x: (1, x, x ** 2)) \
        .reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))
    N = parameters[0]
    mean = parameters[1] / N
    std = (parameters[2] / N - mean ** 2) ** 0.5

    print(f'mean: {mean:.3f}, std: {std:.3f}')
    not_outliers = rdd \
        .map(lambda x: (x, (x-mean) / std)) \
        .filter(lambda x: x[1] <= 3) \
        .keys() \
        .collect()
    return not_outliers


def main():
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("task2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    arr = [-1, 1, 3, 2, 2, 150, 1, 2, 3, 2, 2, 1, 1, 1, -100, 2, 2, 3, 4, 1, 1, 3, 4]
    arr = list(map(lambda x: float(x), arr))
    rdd = sc.parallelize(arr)

    # find statistics and items inside 3*sigma gap
    print(findNotOutliers(sc, rdd))

    # test
    print(Statistics.kolmogorovSmirnovTest(rdd, "norm"))

    spark.stop()


if __name__ == '__main__':
    main()
