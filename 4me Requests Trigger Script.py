# Databricks notebook source

#Pull raw request data from api
import requests
import json
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import *
import re

id = []
sourceID = []
subject = []
category = []
impact = []
status = []
next_target_at = []
completed_at = []
team = []
member = []
grouped_into = []
service_instance = []
created_at = []
updated_at = []
#tags = []
nodeID = []

def get_requests(url, payload, headers):
  response = requests.request("GET", url, headers=headers, data=payload)
  response_text = response.text[1:-1]
  data = json.loads(response.text)
  
  for i in data:
    #for key, value in i.items():
    id.append(i["id"])
    sourceID.append(i["sourceID"])
    subject.append(i["subject"])
    category.append(i["category"])
    impact.append(i["impact"])
    status.append(i["status"])
    next_target_at.append(i["next_target_at"])
    completed_at.append(i["completed_at"])
    team.append(i["team"])
    member.append(i["member"])
    grouped_into.append(i["grouped_into"])
    service_instance.append(i["service_instance"])
    created_at.append(i["created_at"])
    updated_at.append(i["updated_at"])
#    tags.append(i["tags"])
    nodeID.append(i["nodeID"])
  
  response_headers = response.headers
  print(response.headers['X-Pagination-Current-Page'])
  return response_headers

url_requests = 'https://api.4me.com/v1/requests?per_page=100?rel="first"'

payload={}
first_headers = {
  'X-4me-Account': 'kgpco-it',
  'Authorization': 'Bearer TPLXmxFxlVN7ImkNn94kOtrY4TwUImn0kqH8PHx2JlGOgke4PWBc1arNsNDjjFiBTz2VCeJLjpxDvaYMuye4RiJEs73weFM9epVkeqbgaomsnIHm'
}
next_headers = {
  'X-4me-Account': 'kgpco-it',
  'Authorization': 'Bearer TPLXmxFxlVN7ImkNn94kOtrY4TwUImn0kqH8PHx2JlGOgke4PWBc1arNsNDjjFiBTz2VCeJLjpxDvaYMuye4RiJEs73weFM9epVkeqbgaomsnIHm',
  'rel': 'next'
}
response_headers = get_requests(url_requests, payload, first_headers)
url = response_headers["link"]
print(url)
formatted_url = url[url.index('<') + 1:url.index('>')]
print(formatted_url)

stop = False
while (not stop):
  response_headers = get_requests(formatted_url, payload, next_headers)
  link = response_headers["link"]
  index = link.find("next")
  #'link' param in response header can have links for first, next, and prev. If next is not found, stop.
  if index != -1:
    first, prev, n = link.split(",")
    #Removing '<>' characters
    formatted_url = n[n.index('<') + 1:n.index('>')]
  else:
    formatted_url = link[link.index('<') + 1:link.index('>')]
    stop = True
  print(formatted_url)


requests_dict = {
    "id": id,
    "sourceID": sourceID,
    "subject": subject,
    "category": category,
    "impact": impact,
    "status": status,
    "next_target_at": next_target_at,
    "completed_at": completed_at,
    "team": team,
    "member": member,
    "grouped_into": grouped_into,
    "service_instance": service_instance,
    "created_at": created_at,
    "updated_at": updated_at,
#    "tags": tags,
    "nodeID": nodeID,
  }

pdf = pd.DataFrame(requests_dict)
  
spark = SparkSession.builder.getOrCreate()
schema = StructType([
           StructField("id", StringType(), True),
           StructField("sourceID", StringType(), True),
           StructField("subject", StringType(), True),
           StructField("category", StringType(), True),
           StructField("impact", StringType(), True),
           StructField("status", StringType(), True),
           StructField("next_target_at", StringType(), True),
           StructField("completed_at", StringType(), True),
           StructField("team", StringType(), True),
           StructField("member", StringType(), True),
           StructField("grouped_into", StringType(), True),
           StructField("service_instance", StringType(), True),
           StructField("created_at", StringType(), True),
           StructField("updated_at", StringType(), True),
#           StructField("tags", StringType(), True),
           StructField("nodeID", StringType(), True),
         ])
  
df = spark.createDataFrame(pdf, schema)

# Register the dataframe as a temporary view
df.createOrReplaceTempView("4me_completed_requests_data")
df.write.option("mergeSchema", "true").mode("append").saveAsTable("4me_request_data")

# COMMAND ----------

# SQL select statement in PySpark
query = """
CREATE OR REPLACE TABLE web_analytics.default.weekdays AS
WITH weekdays AS (
  SELECT DISTINCT
    team, member, category, impact, subject, 
    created_at,
    completed_at,
    DATEDIFF(completed_at, created_at) AS DifferenceInDays
  FROM
    4me_request_data
  WHERE
    status = "completed"
    AND impact != "NULL"
)
SELECT
  CASE WHEN DAYOFWEEK(created_at) > DAYOFWEEK(completed_at) then DifferenceInDays -2
           ELSE DifferenceInDays END as num_days, created_at, completed_at, team, member, category, impact, subject
FROM
  weekdays;
"""

result_df = spark.sql(query)
display(result_df)

# COMMAND ----------

from pyspark.sql.functions import current_date, date_sub

# Clear the database
spark.sql("DROP TABLE IF EXISTS requests_last_year")

# Create and fill the database with entries from 'weekdays' where created_at is between today and one year ago
query = """
CREATE TABLE requests_last_year AS
SELECT DISTINCT team, member, category, impact, subject, 
    CAST(created_at AS DATE) AS created_at,
    CAST(completed_at AS DATE) AS completed_at,
    num_days
FROM web_analytics.default.weekdays
WHERE created_at BETWEEN date_sub(current_date(), 365) AND current_date()
"""

spark.sql(query)

# COMMAND ----------

# MAGIC %pip install nltk
# MAGIC
# MAGIC from pyspark.sql.functions import explode, split, col, lower, collect_list, concat_ws, length
# MAGIC import nltk
# MAGIC from nltk.corpus import wordnet as wn
# MAGIC
# MAGIC # Ensure NLTK data is downloaded
# MAGIC nltk.download('averaged_perceptron_tagger')
# MAGIC nltk.download('wordnet')
# MAGIC
# MAGIC # Function to check if a word is a noun
# MAGIC def is_noun(word):
# MAGIC     synsets = wn.synsets(word)
# MAGIC     if not synsets:
# MAGIC         return False
# MAGIC     return any(ss.pos() == 'n' for ss in synsets)
# MAGIC
# MAGIC # Load the data from the requests_last_year table
# MAGIC requests_df = spark.table("requests_last_year")
# MAGIC
# MAGIC # Filter rows where num_days is greater than 4
# MAGIC filtered_requests_df = requests_df.filter(col("num_days") > 4)
# MAGIC
# MAGIC # Split the subject into words, convert to lowercase, and explode into individual words
# MAGIC words_df = filtered_requests_df.withColumn("word", explode(split(lower(col("subject")), "\\s+")))
# MAGIC
# MAGIC # Filter only nouns using a Python list comprehension
# MAGIC words = words_df.select("word").distinct().rdd.flatMap(lambda x: x).collect()
# MAGIC nouns = [word for word in words if is_noun(word)]
# MAGIC
# MAGIC # Filter the DataFrame to keep only nouns and words longer than one character
# MAGIC nouns_df = words_df.filter((col("word").isin(nouns)) & (length(col("word")) > 1))
# MAGIC
# MAGIC # Group by team and word, then count occurrences
# MAGIC word_count_df = nouns_df.groupBy("team", "word").count()
# MAGIC
# MAGIC # Find the top 5 most common words for each team
# MAGIC from pyspark.sql.window import Window
# MAGIC from pyspark.sql.functions import row_number
# MAGIC
# MAGIC window_spec = Window.partitionBy("team").orderBy(col("count").desc())
# MAGIC most_common_words_df = word_count_df.withColumn("row_number", row_number().over(window_spec)).filter(col("row_number") <= 5).drop("row_number")
# MAGIC
# MAGIC # Aggregate the top 5 words into a comma-separated list for each team
# MAGIC result_df = most_common_words_df.groupBy("team").agg(concat_ws(", ", collect_list("word")).alias("words"))
# MAGIC
# MAGIC display(result_df)

# COMMAND ----------

result_df.write.mode("overwrite").saveAsTable("team_aggr_analytics")
