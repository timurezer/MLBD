import os
from pyspark.sql import SparkSession


def set_up_env():
    os.environ["SPARK_HOME"] = "/home/timurezer/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/timurezer/anaconda3/envs/mlbd/bin/python"

def main():
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("task2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    # sets of languages
    jvmLanguages = sc.parallelize(["Scala", "Java", "Groovy", "Kotlin", "Ceylon"])
    functionalLanguages = sc.parallelize(["Scala", "Kotlin", "JavaScript", "Haskell", "Python"])
    webLanguages = sc.parallelize(["PHP", "Ruby", "Perl", "JavaScript", "Python"])
    mlLanguages = sc.parallelize(["JavaScript", "Python", "Scala"])

    # question 1
    set1 = jvmLanguages.intersection(mlLanguages).collect()
    print(f'JVM and ML languages: {", ".join(set1)}')

    # question 2
    set2 = webLanguages.subtract(functionalLanguages).collect()
    print(f'WEB and not Func: {", ".join(set2)}')

    # question 3
    set3 = jvmLanguages.union(functionalLanguages).distinct().collect()
    print(f'JVM with Func: {", ".join(set3)}')

    spark.stop()


if __name__ == '__main__':
    main()

