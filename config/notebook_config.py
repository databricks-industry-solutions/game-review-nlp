# Databricks notebook source
# MAGIC %md
# MAGIC ## Install and import necessary packages

# COMMAND ----------

# MAGIC %pip install demoji spark-nlp==4.2.0 git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

import os
import requests

#Unicodedata module provides access to the Unicode Character Database (UCD)
import unicodedata as uni

#Demoji helps to process emojis
import demoji

#re for regular expressions
import re

from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.sql.functions import col,udf
from pyspark.sql.types import StringType

import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *

#libraries for clustering
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set variables and paths

# COMMAND ----------

useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username = useremail.split('@')[0]
username_sql = re.sub('\W', '_', username)
tmpdir = f"/dbfs/tmp/{username}/"
tmpdir_dbfs = f"/tmp/{username}"
catalog_name = f"gaming_nlp_{username_sql}"
database_name = "game_analysis"
database_location = f"{tmpdir}gaming_nlp"

# COMMAND ----------

os.environ['tmpdir'] = tmpdir

# COMMAND ----------

# MAGIC %md
# MAGIC If you wish to use a different game, you can specify it in the notebook_config, by looking up your game on the Steam website and getting the ID from the URL. If no game is specified, we will default to New World.
# MAGIC
# MAGIC If you use one of the games in the dictionary in the 01 notebook, specifying the <i>game_name</i> is sufficient. Otherwise, please specify both <i>user_game_id</i> and <i>game_name</i>. <i>game_name</i> is used downstream to name the output table.

# COMMAND ----------

user_game_id = ''

# Default to New World, this variable is also used to name the output table downstream
game_name = 'New World'

# Replace blanks in the name with underscore
game_name_sub = re.sub(r"\s", "_", f"{game_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC <i>author_cols</i> specifies the columns from steam that have metadata about the reviewer/author, we use this list throughout the notbooks.

# COMMAND ----------

author_cols = ["author_steamid", "author_num_reviews", "author_num_games_owned", \
               "author_last_played", "author_playtime_forever", \
               "author_playtime_at_review", "author_playtime_at_last_two_weeks" \
              ]
