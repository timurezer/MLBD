import os

from pyspark.sql import SparkSession

def save_numbers(sc, partitions, path):
    numbers = range(100000)
    rdd = sc.parallelize(numbers, partitions)
    rdd.saveAsTextFile(path)

# def read_numbers(sc, path):
#     files = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
#     print(files)
#     rdd = sc.textFile(','.join(files))
#     return rdd

def main():
    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("RDD_Intro") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    path = './tmp'
    save_numbers(sc, 10, path)
    # rdd = read_numbers(sc, path)
    # print(rdd.getNumPartitions())
    spark.stop()

if __name__ == '__main__':
    main()
