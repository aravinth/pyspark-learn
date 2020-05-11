# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Model, PipelineModel
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, MinHashLSH

from pyspark.sql.types import *
from pyspark.sql.functions import col

'''
This is the scalable version for matching the sentences between two
big datasets using the hash value of the sentences.
'''


'''
---------------------------------------
SPARK SESSION CREATION
---------------------------------------
'''
spark = SparkSession \
    .builder \
    .appName("Sentence_Match_Tool") \
    .getOrCreate()


spark.conf.set("spark.dynamicAllocation.enabled", "true")
spark.conf.set("spark.executor.cores", 6)
spark.conf.set("spark.dynamicAllocation.minExecutors","3")
spark.conf.set("spark.dynamicAllocation.maxExecutors","6")


LOG4JLOGGER = spark.sparkContext._jvm.org.apache.log4j
LOGGER = LOG4JLOGGER.LogManager.getLogger(__name__)
LOGGER.info("-------------------------------------------------------")
LOGGER.info("Starting the Sentence Matcher...")
LOGGER.info("-------------------------------------------------------")

# Define the Inputs
# Schema : <filename, sentences>
PDF_INPUT_FILE="/user/root/dataset1.csv"
# Schema : <sentence>
IK_INPUT_FILE="/user/root/dataset2.csv"

# Schema of the files
SCHEMA_PDF_INPUT_FILE = StructType([StructField("filename", StringType(), True), StructField("sentence", StringType(), True)])
INPUT_FILE_DF    = spark.read.format("com.databricks.spark.csv") \
                  .option("delimiter", '\t') \
                  .option("quote", "\"") \
                  .option("escape", "\"") \
                  .option("header", "false") \
                  .schema(SCHEMA_PDF_INPUT_FILE) \
                  .load(PDF_INPUT_FILE)
DF_INPUT_FILE_CLEAN    = INPUT_FILE_DF.filter("sentence rlike '[A-Z,a-z]'")

SCHEMA_IK_INPUT_FILE = StructType([StructField("sentence", StringType(), True)])
IK_FILE_DF = spark.read.format("com.databricks.spark.csv") \
                  .option("delimiter", ',') \
                  .option("quote", "\"") \
                  .option("escape", "\"") \
                  .option("header", "false") \
                  .schema(SCHEMA_IK_INPUT_FILE) \
                  .load(IK_INPUT_FILE)
IK_FILE_DF_CLEAN    = IK_FILE_DF.filter("sentence rlike '[A-Z,a-z]'")

# SQL Operations
DF_INPUT_FILE_CLEAN.createOrReplaceTempView('DF_INPUT_FILE_CLEAN')
IK_FILE_DF_CLEAN.createOrReplaceTempView('IK_FILE_DF_CLEAN')

matched_df = spark.sql('select a.sentence as pdf_sentence, b.sentence as ik_sentence, hash(a.sentence) as hash_val, filename from DF_INPUT_FILE_CLEAN a, IK_FILE_DF_CLEAN b where hash(a.sentence)=hash(b.sentence)')

matched_df.write.csv('/user/root/exact-match-score/output-temp')

# matched_df.coalesce(1).write \
#         .format("com.databricks.spark.csv") \
#         .option("header", "true") \
#         .mode("overwrite") \
#         .save("/user/root/exact-match-score/output-temp/")

