# Databricks notebook source
# MAGIC %md
# MAGIC # Post-Processing: Author Clustering and Output Results

# COMMAND ----------

# MAGIC %md
# MAGIC The analysis notebook outputs a results table to Unity Catalog. In this notebook we will create author clusters to better visualize NLP results in the dashboard.

# COMMAND ----------

# Create a list of author columns
author_cols = ["author_steamid", "author_num_reviews", "author_num_games_owned", \
               "author_last_played", "author_playtime_forever", \
               "author_playtime_at_review", "author_playtime_at_last_two_weeks" \
              ]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Read Data from Unity Catalog

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG gaming_nlp

# COMMAND ----------

database_name = "new_world"
table_name = "results"

df = spark.sql(f"SELECT * FROM {database_name}.{table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Profile Author Columns

# COMMAND ----------

# MAGIC %md 
# MAGIC One way to quickly visualize variable distributions in Databricks is to display the DataFrame and then add a Data Profile. Doing this here to see the Author variable distribution. We also add some specific visualizations for <i>Author Number of Reviews</i> and <i>Playtime at Review</i>.

# COMMAND ----------

display(df[[*author_cols]].limit(10000))

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Perform K-means Clustering

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have profiled and understand some of the author columns, we can use the numeric columns to cluster the authors using K-means Clustering.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# Columns to cluster on
author_cols_num = [m for m in author_cols if m not in ["author_steamid","author_last_played"]]

# Vectorize the columns of interest
vec_assembler = VectorAssembler(inputCols=author_cols_num, outputCol="features_vector")
output = vec_assembler.transform(df).select('*', 'features_vector')

# Standardize features
std_scaler = StandardScaler(inputCol='features_vector', outputCol='features', withStd=True, withMean=True)
scaler_model = std_scaler.fit(output)
output = scaler_model.transform(output).select('*').drop('features_vector')

# Run K-means clustering
kmeans = KMeans(k=3, seed=1)
model = kmeans.fit(output)

# Predict the clusters of each data point
df_clustered = model.transform(output)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now take a look at some of the results. The cluster number gets output in a variable named <i>prediction</i>.

# COMMAND ----------

display(df_clustered)

# COMMAND ----------

# MAGIC %md
# MAGIC To bettter understand the clusters, it is useful to plot them. You can change <i>col1</i> and <i>col2</i> to get different views and better understand the clusters.

# COMMAND ----------

import matplotlib.pyplot as plt

col1 = 4
col2 = 2

# Get the cluster centers
centers = model.clusterCenters()

# Extract the features and predicted clusters
features = output.select('features').rdd.map(lambda row: row[0]).collect()
predictions = df_clustered.select('prediction').rdd.map(lambda row: row[0]).collect()

# Plot the data points colored by their predicted cluster
plt.scatter([f[col1] for f in features], [f[col2] for f in features], c=predictions)
plt.title('K-means Clustering Results')
plt.xlabel(author_cols_num[col1])
plt.ylabel(author_cols_num[col2])

# Plot the cluster centers
for center in centers:
    plt.scatter(center[col1], center[col2], marker='x', s=200, linewidths=3, color='black')

# Show the plot
display(plt.show())

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's look at the average <i>author_playtime_forever</i> and <i>author_playtime_at_last_two_weeks</i> to better understand and give meaning to the clusters.

# COMMAND ----------

from pyspark.sql.functions import avg

# Select the desired columns and group by prediction
df_grouped = df_clustered.select('prediction', 'author_playtime_forever', 'author_playtime_at_last_two_weeks') \
                        .groupBy('prediction')

# Calculate the averages of the playtime columns
df_avg = df_grouped.agg(avg('author_playtime_forever'), avg('author_playtime_at_last_two_weeks'))

# Show the resulting dataframe
display(df_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC It appears that cluster 0 has low playtime recently and forever, cluster 1 has high playtime in the last two weeks but medium playtime forever, and cluster 2 has high playtime forever and medium-low time in the last two weeks. We can give these more descriptive names for the dashboard.

# COMMAND ----------

# TO DO: Add descriptive names.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.  Save results to a Delta Table and create a DB SQL dashboard.

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DROP TABLE IF EXISTS new_world.results_clustered

# COMMAND ----------

database_name = "new_world"
table_name = "results_clustered"

# Use "delta" format for Unity Catalog
df_clustered.write \
    .format("delta") \
    .saveAsTable(f"{database_name}.{table_name}")
