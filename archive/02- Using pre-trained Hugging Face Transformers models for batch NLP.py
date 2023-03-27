# Databricks notebook source
# MAGIC %md
# MAGIC # Use a Hugging Face Transformers model for batch NLP
# MAGIC This notebook shows how to use a pre-trained [🤗 Transformers](https://huggingface.co/docs/transformers/index) model to perform NLP tasks easily on Spark. It also highlights best practices on getting and understanding performance. 
# MAGIC 
# MAGIC This example shows using a pre-trained summarization [pipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines) directly as a UDF and  an MLflow model to summarize Wikipedia articles.
# MAGIC 
# MAGIC ## Cluster setup
# MAGIC For this notebook, Databricks recommends a multi-machine, multi-GPU cluster, such as an 8 worker `p3.2xlarge` cluster on AWS or `NC6s_v3` on Azure using Databricks Runtime ML with GPU versions 10.4 or greater. 
# MAGIC 
# MAGIC The recommended cluster configuration in this notebook takes about 10 minutes to run. GPU auto-assignment does not work for single-node clusters, so Databricks recommends performing GPU inference on clusters with separate drivers and workers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up parameters
# MAGIC 
# MAGIC For this example, you can use the following parameters:
# MAGIC 
# MAGIC - `output_schema` is the database schema to write data to.
# MAGIC - `output_table` is an output table. Which you can delete with the last command of this notebook.
# MAGIC - `number_articles` is the number of articles to sample and summarize.

# COMMAND ----------

output_schema = "tutorials"
output_table = "wikipedia_summaries"
number_articles = 1024

# COMMAND ----------

# MAGIC %md
# MAGIC # Data loading
# MAGIC 
# MAGIC This notebook uses a sample of Wikipedia articles as the dataset.

# COMMAND ----------

df = spark.read.parquet("/databricks-datasets/wikipedia-datasets/data-001/en_wikipedia/articles-only-parquet").select("title", "text")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC It's important for resource utilization that there be enough partitions of the DataFrame to fully utilize the parallelization available in the cluster. Generally, some multiple of the number of GPUs on your workers (for GPU clusters) or number of cores across the workers in your cluster (for CPU clusters) works well in practice. This helps get more balanced resource utilization across the cluster. 
# MAGIC 
# MAGIC If you do not repartition the data after applying a limit, you may wind up underutilizing your cluster. For example, if only one partition is required to limit the dataset, Spark sends that single partition to one executor.

# COMMAND ----------

sample_imbalanced = df.limit(number_articles)
sample = sample_imbalanced.repartition(32).persist()
sample.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Using a pipeline in a Pandas UDF

# COMMAND ----------

# MAGIC %md
# MAGIC The next command loads the `transformers` pipeline for summarization using the `facebook/bart-large-cnn` model. 
# MAGIC 
# MAGIC [Pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines) conveniently wrap best practices for certain tasks, bundling together tokenizers and models. They can also help with batching data sent to the GPU, so that you can perform inference on multiple items at a time. Setting the `device` to 0 causes the pipeline to use the GPU for processing. You can use this setting reliably even if you have multiple GPUs on each machine in your Spark cluster. Spark automatically reassigns GPUs to the workers.
# MAGIC 
# MAGIC You can also directly load tokenizers and models if needed; you would just need to reference and invoke them directly in the UDF.

# COMMAND ----------

from transformers import pipeline
import torch
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks recommends using [Pandas UDFs](https://www.databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html) to apply the pipeline. Spark sends batches of data to Pandas UDFs and uses `arrow` for data conversion. Receiving a batch in the UDF allows you to batch operations in the pipeline. Note that the `batch_size` in the pipeline is unlikely to be performant with the default batch size that Spark sends the UDF. Databricks recommends trying various batch sizes for the pipeline on your cluster to find the best performance. Read more about pipeline batching in [Hugging Face documentation](https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching).
# MAGIC 
# MAGIC You can view GPU utilization on the cluster by navigating to the [live Ganglia metrics](https://docs.databricks.com/clusters/clusters-manage.html#ganglia-metrics), clicking into a particular worker, and viewing the GPU metrics section for that worker. 
# MAGIC 
# MAGIC Wrapping the pipeline with [tqdm](https://tqdm.github.io/) allows you to view progress of a particular task. Navigate into the [task details page](https://docs.databricks.com/clusters/debugging-spark-ui.html#task-details-page) and view the `stderr` logs.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from tqdm.auto import tqdm

@pandas_udf('string')
def summarize_batch_udf(texts: pd.Series) -> pd.Series:
  pipe = tqdm(summarizer(texts.to_list(), truncation=True, batch_size=8), total=len(texts), miniters=10)
  summaries = [summary['summary_text'] for summary in pipe]
  return pd.Series(summaries)

# COMMAND ----------

# MAGIC %md
# MAGIC Using the UDF is identical to using other UDFs on Spark.  For example, you can use it in a `select` statement to create a column with the results of the model inference.

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {output_schema}")
summaries = sample.select(sample.title, sample.text, summarize_batch_udf(sample.text).alias("summary"))
summaries.write.saveAsTable(f"{output_schema}.{output_table}", mode="overwrite")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {output_schema}.{output_table} LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC #MLflow wrapping
# MAGIC Storing a pre-trained model as an MLflow model makes it even easier to deploy a model for batch or real-time inference. This also allows model versioning through the Model Registry, and simplifies model loading code for your inference workloads. 
# MAGIC 
# MAGIC The first step is to create a custom model for your pipeline, which encapsulates loading the model, initializing the GPU usage, and inference function. 

# COMMAND ----------

import mlflow

class SummarizationPipelineModel(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    device = 0 if torch.cuda.is_available() else -1
    self.pipeline = pipeline("summarization", context.artifacts["pipeline"], device=device)
    
  def predict(self, context, model_input): 
    texts = model_input.iloc[:,0].to_list() # get the first column
    pipe = tqdm(self.pipeline(texts, truncation=True, batch_size=8), total=len(texts), miniters=10)
    summaries = [summary['summary_text'] for summary in pipe]
    return pd.Series(summaries)

# COMMAND ----------

# MAGIC %md
# MAGIC The code closely parallels the code for creating and using a Pandas UDF demonstrated above. One difference is that the pipeline is loaded from a file made available to the MLflow model’s context. This is provided to MLflow when logging a model. 🤗 Transformers pipelines make it easy to save the model to a local file on the driver, which is then passed into the `log_model` function for the MLflow `pyfunc` interfaces. 

# COMMAND ----------

pipeline_path = "./pipeline"

summarizer.save_pretrained(pipeline_path)
with mlflow.start_run() as run:        
  mlflow.pyfunc.log_model(artifacts={'pipeline': pipeline_path}, artifact_path="summarization_model", python_model=SummarizationPipelineModel())

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow scoring
# MAGIC MLflow provides an easy interface to load any logged or registered model into a spark UDF. You can look up a model URI from the Model Registry or logged experiment run UI. 

# COMMAND ----------

import mlflow
logged_model_uri = f"runs:/{run.info.run_id}/summarization_model"

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model_uri, result_type='string')

summaries = sample.select(sample.title, sample.text, loaded_model(sample.text).alias("summary"))
summaries.write.saveAsTable(f"{output_schema}.{output_table}", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleanup
# MAGIC Remove the output table this notebook writes results to.

# COMMAND ----------

spark.sql(f"DROP TABLE {output_schema}.{output_table}")
