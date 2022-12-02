import os
from pyspark.sql import SparkSession
from math import log
import shutil

PATH = 'RDD/task1/tmp'

def set_up_env():
    os.environ["SPARK_HOME"] = "/home/timurezer/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/timurezer/anaconda3/envs/mlbd/bin/python"

def save_numbers(sc, partitions, path):
    '''
    create rdd with stack of numbers and save them into path
    '''
    numbers = range(1, 100000)
    rdd = sc.parallelize(numbers, partitions)
    # delete folder to not obtain errors
    if os.path.isdir(path):
        shutil.rmtree(path)
    rdd.saveAsTextFile(path)

def main():
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("task1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    # create and save numbers
    save_numbers(sc, 10, PATH)

    # load numbers
    rdd = sc.textFile(PATH)

    # create keys and values
    rdd = rdd.map(lambda x: (int(x) % 100, log(float(x))))

    # filter and group by keys
    result = rdd \
        .filter(lambda x: (int(10 * x[1]) % 10) % 2 == 0) \
        .groupByKey() \
        .mapValues(len) \
        .collect()

    # print to output
    for key, value in result:
        print(f'{key}: {value}')

    spark.stop()


if __name__ == '__main__':
    main()

