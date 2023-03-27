# Databricks notebook source
# DBTITLE 1,Database settings
database = "game_review"

spark.sql(f"create database if not exists {database}")
spark.sql(f"use {database}")

# COMMAND ----------

# DBTITLE 1,mlflow settings - This may not be used
import mlflow
model_name = "game_review"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f'/Users/{username}/{model_name}')

# COMMAND ----------

# DBTITLE 1,Streaming checkpoint location - This may not be used
checkpoint_path = f"/dbfs/tmp/{username}/game-review/checkpoints"

# COMMAND ----------

# DBTITLE 1,Persist transformers cache in DBFS
import os
os.environ['TRANSFORMERS_CACHE'] = f"/dbfs/tmp/{username}/cache/hf"
