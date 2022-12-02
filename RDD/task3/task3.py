import os
from pyspark.sql import SparkSession


def set_up_env():
    os.environ["SPARK_HOME"] = "/home/timurezer/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/timurezer/anaconda3/envs/mlbd/bin/python"

def count_salaries(sc, salaries: list):
    result = sc.parallelize(salaries) \
        .map(lambda x: x.split()) \
        .map(lambda x: (x[0], float(x[1]))) \
        .groupByKey() \
        .mapValues(lambda x: sum(x) / len(x)) \
        .collect()

    for name, value in result:
        print(f'{name}: {value}')

def main():
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("task3") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    salaries = [
        "John 1900 January",
        "Mary 2000 January",
        "John 1800 February",
        "John 1000 March",
        "Mary 1500 February",
        "Mary 2900 March",
        "Mary 1600 April",
        "John 2800 April"
    ]

    count_salaries(sc, salaries)
    spark.stop()


if __name__ == '__main__':
    main()

