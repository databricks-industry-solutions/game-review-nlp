# Databricks notebook source
# MAGIC %md
# MAGIC # Sentiment Analysis from [Steam](https://store.steampowered.com/) Game Reviews

# COMMAND ----------

# MAGIC %md
# MAGIC ![Steam Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Steam_icon_logo.svg/240px-Steam_icon_logo.svg.png)

# COMMAND ----------

# MAGIC %md
# MAGIC <b>To do:</b><br><br>Still need to save results from 02 notebook in a way that can be picked up by this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.  Save results to a Delta Table and create a DB SQL dashboard.

# COMMAND ----------

results_path = "/Users/anil.joshi@databricks.com/Player-Feedback-Accelerator/NewWorld"

(finalResults
 .write
 .format("delta")
 .mode("overwrite")
 .save(results_path)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS `PFADatabase`;
# MAGIC USE PFADatabase;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE PFA_Final_Results_NewWorld
# MAGIC (
# MAGIC    reviewsCleaned  STRING,
# MAGIC    chunk   STRING,
# MAGIC    ner_label  STRING
# MAGIC  )
# MAGIC  LOCATION "/Users/anil.joshi@databricks.com/Player-Feedback-Accelerator/NewWorld"
