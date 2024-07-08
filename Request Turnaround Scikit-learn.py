# Databricks notebook source
# MAGIC %pip install nltk
# MAGIC from sklearn.preprocessing import LabelEncoder
# MAGIC from pyspark.sql.functions import Column
# MAGIC import pyspark
# MAGIC from pyspark.sql.functions import lit
# MAGIC from pyspark.sql.types import Row
# MAGIC from pyspark.sql.types import StructType, StructField, StringType, IntegerType
# MAGIC from pyspark.sql import SparkSession
# MAGIC from pyspark.sql.types import *
# MAGIC import pandas as pd
# MAGIC import nltk
# MAGIC
# MAGIC nltk.download('all')
# MAGIC
# MAGIC import re
# MAGIC from nltk.corpus import stopwords
# MAGIC from nltk.stem import WordNetLemmatizer
# MAGIC
# MAGIC df_requests = spark.sql('WITH weekdays AS (SELECT DISTINCT team, member, category, impact, subject, created_at, completed_at, DATEDIFF(completed_at, created_at) AS DifferenceInDays FROM 4me_request_data WHERE status = "completed" AND impact != "NULL") SELECT CASE WHEN DAYOFWEEK (created_at) > DAYOFWEEK(completed_at) then DifferenceInDays -2 ELSE DifferenceInDays END as num_days, created_at, team, member, category, impact, subject FROM weekdays;').toPandas()
# MAGIC
# MAGIC df_requests.head()
# MAGIC #print the size of df_requests
# MAGIC print(df_requests.shape)

# COMMAND ----------

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk

# Function to label subject and object in sentences
def label_subject_object(sentence):
    tagged_words = pos_tag(word_tokenize(sentence))
    named_entities = ne_chunk(tagged_words)
    labels = []
    for tree in named_entities:
        if hasattr(tree, 'label'):
            word = ' '.join(c[0] for c in tree)
            labels.append((word, tree.label()))
    subjects = [label[0] for label in labels if label[1] == 'PERSON' or label[1] == 'ORGANIZATION']
    objects = [label[0] for label in labels if label[1] == 'GPE' or label[1] == 'LOCATION']
    return {'subjects': subjects, 'objects': objects}

# Apply the function to the 'subject' column of df_requests
df_requests['subject_object_labels'] = df_requests['subject'].apply(label_subject_object)

# Convert the updated DataFrame back to a Spark DataFrame
df_requests_spark = spark.createDataFrame(df_requests)

display(df_requests_spark)

# COMMAND ----------

from pyspark.sql.functions import concat_ws

# Concatenate 'subjects' and 'objects' in 'subject_object_labels' into comma-delimited strings
df_concatenated = df_requests_spark.withColumn("subject_part_of_speech", concat_ws(", ", "subject_object_labels.subjects")) \
                                   .withColumn("objects", concat_ws(", ", "subject_object_labels.objects")) \
                                   .drop("subject_object_labels")

display(df_concatenated)

# COMMAND ----------

from pyspark.sql.functions import when, col

# Fill empty 'subject_part_of_speech' and 'objects' columns with 'NONE'
df_filled = df_concatenated.withColumn("subject_part_of_speech", when(col("subject_part_of_speech") == "", "NONE").otherwise(col("subject_part_of_speech"))) \
                           .withColumn("objects", when(col("objects") == "", "NONE").otherwise(col("objects")))

display(df_filled)

# COMMAND ----------

from pyspark.sql.functions import split, explode, expr

# Split the comma-separated values into arrays
df_split = df_filled.withColumn("subject_part_of_speech_array", split(col("subject_part_of_speech"), ", ")) \
                    .withColumn("objects_array", split(col("objects"), ", "))

# Explode the arrays into separate rows
df_exploded = df_split.withColumn("subject_part_of_speech_exploded", explode(col("subject_part_of_speech_array"))) \
                     .withColumn("objects_exploded", explode(col("objects_array")))



display(df_exploded)

# COMMAND ----------

from pyspark.sql.functions import col, expr

# Create new columns for subject_1, subject_2, subject_3, subject_4, subject_5
df_subjects = df_exploded.withColumn("subject_1", col("subject_part_of_speech_exploded")) \
                          .withColumn("subject_2", expr("subject_part_of_speech_array[1]")) \
                          .withColumn("subject_3", expr("subject_part_of_speech_array[2]")) \
                          .withColumn("subject_4", expr("subject_part_of_speech_array[3]")) \
                          .withColumn("subject_5", expr("subject_part_of_speech_array[4]"))

# Create new columns for object_1, object_2, object_3, object_4, object_5
df_objects = df_subjects.withColumn("object_1", col("objects_exploded")) \
                        .withColumn("object_2", expr("objects_array[1]")) \
                        .withColumn("object_3", expr("objects_array[2]")) \
                        .withColumn("object_4", expr("objects_array[3]")) \
                        .withColumn("object_5", expr("objects_array[4]"))

# Select the relevant columns
df_combined = df_objects.select("num_days", "created_at", "team", "member", "category", "impact",
                                "subject_1", "subject_2", "subject_3", "subject_4", "subject_5",
                                "object_1", "object_2", "object_3", "object_4", "object_5")

display(df_combined)
display(df_combined.count())

# COMMAND ----------

from pyspark.sql.functions import coalesce, lit

# Fill null values in subject_1 through subject_5 with "NONE"
df_filled_subjects = df_combined.withColumn("subject_1", coalesce(col("subject_1"), lit("NONE"))) \
                               .withColumn("subject_2", coalesce(col("subject_2"), lit("NONE"))) \
                               .withColumn("subject_3", coalesce(col("subject_3"), lit("NONE"))) \
                               .withColumn("subject_4", coalesce(col("subject_4"), lit("NONE"))) \
                               .withColumn("subject_5", coalesce(col("subject_5"), lit("NONE")))

# Fill null values in object_1 through object_5 with "NONE"
df_filled_objects = df_filled_subjects.withColumn("object_1", coalesce(col("object_1"), lit("NONE"))) \
                                      .withColumn("object_2", coalesce(col("object_2"), lit("NONE"))) \
                                      .withColumn("object_3", coalesce(col("object_3"), lit("NONE"))) \
                                      .withColumn("object_4", coalesce(col("object_4"), lit("NONE"))) \
                                      .withColumn("object_5", coalesce(col("object_5"), lit("NONE")))

display(df_filled_objects.select("num_days", "created_at", "team", "member", "category", "impact", 
                                 "subject_1", "subject_2", "subject_3", "subject_4", "subject_5",
                                 "object_1", "object_2", "object_3", "object_4", "object_5"))

display(df_filled_objects.count())

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# Select the relevant columns for encoding
df_encoded = df_filled_objects.select("num_days", "created_at", "team", "member", "category", "impact", "subject_1", "subject_2", "subject_3", "subject_4", "subject_5",
                                      "object_1", "object_2", "object_3", "object_4", "object_5")

# Loop through each column and encode the values
for col_name in df_encoded.columns:
    if col_name != "num_days":
        indexer = StringIndexer(inputCol=col_name, outputCol=col_name+"_encoded", handleInvalid="skip")
        df_encoded = indexer.fit(df_encoded).transform(df_encoded)

display(df_encoded)
display(df_encoded.count())

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans

# Select only the encoded columns for the vector assembler
encoded_cols = ["created_at_encoded", "team_encoded", "member_encoded", "category_encoded", "impact_encoded", "subject_1_encoded", "subject_2_encoded", "subject_3_encoded", "subject_4_encoded", "subject_5_encoded", "object_1_encoded", "object_2_encoded", "object_3_encoded", "object_4_encoded", "object_5_encoded"]
assembler = VectorAssembler(inputCols=encoded_cols, outputCol="features")
df_assembled = assembler.transform(df_encoded)

# Function to evaluate PCA with different k values
def evaluate_pca_k(df, max_k):
    evaluator = ClusteringEvaluator(featuresCol="pca_features")
    results = []
    for k in range(2, max_k + 1):
        pca = PCA(k=k, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(df)
        df_pca = pca_model.transform(df)
        
        kmeans = KMeans(k=5, seed=42, featuresCol="pca_features")
        model = kmeans.fit(df_pca)
        df_clustered = model.transform(df_pca)
        
        silhouette = evaluator.evaluate(df_clustered)
        results.append((k, silhouette))
    return results

# Evaluate PCA for k values from 2 to 10
pca_results = evaluate_pca_k(df_assembled, 10)

# Find the optimal k with the highest silhouette score
optimal_k = max(pca_results, key=lambda x: x[1])[0]

# Perform PCA with the optimal k
pca = PCA(k=optimal_k, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(df_assembled)
df_pca = pca_model.transform(df_assembled)

display(df_pca)

# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Define a UDF to convert the PCA features to a string
def pca_features_to_string(pca_features):
    return str(pca_features)

pca_features_to_string_udf = udf(pca_features_to_string, StringType())

# Add a new column with the PCA features as a string
df_pca_with_string = df_pca.withColumn("pca_features_string", pca_features_to_string_udf(col("pca_features")))

display(df_pca_with_string)

# COMMAND ----------

from pyspark.sql.functions import split, col, regexp_replace

# Remove brackets from pca_features_string
df_cleaned = df_pca_with_string.withColumn("pca_features_string", regexp_replace(col("pca_features_string"), "[\\[\\]]", ""))

# Split the pca_features_string into separate columns
df_split = df_cleaned.withColumn("Val1", split(col("pca_features_string"), ",").getItem(0).cast("double")) \
                     .withColumn("Val2", split(col("pca_features_string"), ",").getItem(1).cast("double"))

# Select the required columns
df_final = df_split.select("num_days", "Val1", "Val2")

display(df_final)
display(df_final.count())

# COMMAND ----------


import pandas as pd

# Create an empty pandas dataframe
pandas_df = pd.DataFrame()

# Feed the columns from df_final into the pandas dataframe one at a time
pandas_df["num_days"] = df_final.select("num_days").toPandas()["num_days"]
pandas_df["val1"] = df_final.select("val1").toPandas()["val1"]
pandas_df["val2"] = df_final.select("val2").toPandas()["val2"]

# COMMAND ----------

pandas_df.drop_duplicates(inplace=True)
print(pandas_df.shape)

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Split the data into features and target variable
X = pandas_df.drop(columns=["num_days"])
y = pandas_df["num_days"]

gb_model = GradientBoostingRegressor()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [3000, 6000, 10000],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [5, 7, 10]
}

grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create and train the gradient boost regression model with adjusted hyperparameters
gb_model = GradientBoostingRegressor(learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'], n_estimators=best_params['n_estimators'])
gb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5
r2 = gb_model.score(X_test, y_test)

print(rmse)
print(r2)

# COMMAND ----------

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor

# Split the data into features and target variable
X = pandas_df.drop(columns=["num_days"])
y = pandas_df["num_days"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [3000, 6000, 10000],
    'max_depth': [5, 7, 10]
}

# Create the random forest regression model
rf_model = RandomForestRegressor(random_state=42)

# Create the grid search object
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

best_params

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

# Split the data into features and target variable
X = pandas_df.drop(columns=["num_days"])
y = pandas_df["num_days"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the random forest regression model with adjusted hyperparameters
rf_model = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5
r2 = rf_model.score(X_test, y_test)

print(rmse)
print(r2)

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

# Split the data into features and target variable
X = pandas_df.drop(columns=["num_days"])
y = pandas_df["num_days"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

# Create the linear regression model
linear_model = LinearRegression()

# Create the grid search object
grid_search = GridSearchCV(estimator=linear_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

best_params

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# Split the data into features and target variable
X = pandas_df.drop(columns=["num_days"])
y = pandas_df["num_days"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model with adjusted hyperparameters
linear_model = LinearRegression(fit_intercept=best_params['fit_intercept'], normalize=best_params['normalize'])
linear_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5
r2 = linear_model.score(X_test, y_test)

print(rmse)
print(r2)

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# Split the data into features and target variable
X = pandas_df.drop(columns=["num_days"])
y = pandas_df["num_days"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

# Create the ridge regression model
ridge_model = Ridge()

# Create the grid search object
grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

best_params

# COMMAND ----------

from sklearn.linear_model import Ridge

# Split the data into features and target variable
X = pandas_df.drop(columns=["num_days"])
y = pandas_df["num_days"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the ridge regression model with adjusted hyperparameters
ridge_model = Ridge(alpha=best_params['alpha'], fit_intercept=best_params['fit_intercept'], normalize=best_params['normalize'])
ridge_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5
r2 = ridge_model.score(X_test, y_test)

print(rmse)
print(r2)

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Split the data into features and target variable
X = pandas_df.drop(columns=["num_days"])
y = pandas_df["num_days"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'poly_features__degree': [2, 3, 4],
    'linear_regression__alpha': [0.1, 1.0, 10.0],
    'linear_regression__fit_intercept': [True, False],
    'linear_regression__normalize': [True, False]
}

# Create the pipeline
pipeline = Pipeline([('poly_features', PolynomialFeatures()), ('linear_regression', Ridge())])

# Create the grid search object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

best_params

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into features and target variable
X = pandas_df.drop(columns=["num_days"])
y = pandas_df["num_days"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the pipeline with best_params
pipeline = Pipeline([('poly_features', PolynomialFeatures(degree=best_params['poly_features__degree'])), ('linear_regression', Ridge(alpha=best_params['linear_regression__alpha'], fit_intercept=best_params['linear_regression__fit_intercept'], normalize=best_params['linear_regression__normalize']))])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5
r2 = pipeline.score(X_test, y_test)

print(rmse)
print(r2)

# COMMAND ----------

from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

poly = pipeline

# Create the voting regressor
voting_regressor = VotingRegressor(estimators=[('gb', gb_model), ('rf', rf_model), ('ridge', ridge_model), ('poly', poly)])

# Fit the voting regressor to the training data
voting_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = voting_regressor.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5
r2 = voting_regressor.score(X_test, y_test)

print(rmse)
print(r2)
