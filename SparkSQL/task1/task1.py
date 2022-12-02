import os
from pyspark.sql import SparkSession
import shutil

PATH = 'SparkSQL/task1/tmp'

def set_up_env():
    os.environ["SPARK_HOME"] = "/home/timurezer/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/timurezer/anaconda3/envs/mlbd/bin/python"

def count_salaries(sc, salaries: list):
    # create rdd
    rdd = sc.parallelize(salaries) \
        .map(lambda x: x.split()) \
        .map(lambda x: (x[0], float(x[1])))

    # create dataFrame
    columns = ['names', 'salary']
    df = rdd.toDF(columns)
    avg_sal = df.groupBy('names').mean('salary')
    avg_sal.show()
    # delete folder to not obtain errors
    if os.path.isdir(PATH):
        shutil.rmtree(PATH)

    avg_sal.write.json(PATH)


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

