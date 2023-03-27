# Databricks notebook source
# MAGIC %md
# MAGIC Use Databricks Runtime 12.2 ML or higher. 
# MAGIC GPU cluster is not necessary. Use a high-mem type, and set `spark.task.cpus` to 8.
# MAGIC 
# MAGIC Please run the `RUNME` notebook to generate a cluster that satisfies these requirement.

# COMMAND ----------

# MAGIC %pip install -U transformers langchain chromadb pypdf pycryptodome

# COMMAND ----------

# MAGIC %md
# MAGIC Read RFP responses from a CSV file. The desired inputs are 'documents', or chunks of text containing question and answer. This is retrieved along with the source for reference.

# COMMAND ----------

# MAGIC %run ./util/notebook-config

# COMMAND ----------

#from pyspark.sql.functions import concat, lit, col
#
#sig_texts = spark.read.option("header", True).option("multiLine", True).csv("/tmp/sean.owen@databricks.com/sig_answers.csv").\
#  fillna("").\
#  select(concat(lit("SIG "), col("Ques Num")), concat(col("Question/Request"), lit(" "), col("Response"), lit(" "), col("Comments"))).\
#  toDF("source", "text")
#display(sig_texts)

# COMMAND ----------

from pyspark.sql.functions import concat, lit, col

sqrc_texts = spark.read.option("header", True).option("multiLine", True).csv("/tmp/sean.owen@databricks.com/sqrc_answers.csv").\
  fillna("").\
  select(concat(lit("SQRC "), col("id")), concat(col("question"), lit(" "), col("platform-aws-mt_short_answer"), lit(" "), col("platform-aws-mt_detailed_answer"))).\
  toDF("source", "text")
display(sqrc_texts)

# COMMAND ----------

# MAGIC %sh rm -r /dbfs/tmp/sean.owen@databricks.com/langchain/db ; mkdir -p /dbfs/tmp/sean.owen@databricks.com/langchain/db

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

#all_texts = sig_texts.union(sqrc_texts)
# Only use SQRC as it's a superset?
all_texts = sqrc_texts
documents = [Document(page_content=r["text"], metadata={"source": r["source"]}) for r in all_texts.collect()]
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db_persist_path = "/dbfs/tmp/sean.owen@databricks.com/langchain/db"
db = Chroma.from_documents(collection_name="sec_docs", documents=documents, embedding=hf_embed, persist_directory=db_persist_path)

# COMMAND ----------

#from langchain.document_loaders import PyPDFLoader
#
#pdf_loader = PyPDFLoader("/dbfs/tmp/sean.owen@databricks.com/db_2022_soc_2.pdf")
#pdf_pages = pdf_loader.load_and_split()
#db.add_documents(pdf_pages)

# COMMAND ----------

db.similarity_search("dummy") # tickle it to persist metadata (?)
db.persist()

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

#model_name = "bigscience/bloom-560m"
model_name = "bigscience/bloom-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# COMMAND ----------

from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50, early_stopping=True, num_beams=4, num_beam_groups=4, repetition_penalty=1.3)
hf_pipe = HuggingFacePipeline(pipeline=pipe)
qa_chain = load_qa_chain(llm=hf_pipe, chain_type="stuff")

# COMMAND ----------

def do_answer_question(question, db_local, qa_chain_local):
  similar_docs = db_local.similarity_search(question, k=8)
  result = qa_chain_local({"input_documents": similar_docs, "question": question})
  # Hack: Cut off incomplete sentences
  #output_sentences = result["output_text"].split(".")
  #if len(output_sentences) > 1:
  #  output_sentences[-1] = ""
  #result["output_text"] = ".".join(output_sentences)
  return result

def answer_question(question):
  result = do_answer_question(question, db, qa_chain)
  print(result["output_text"])
  for d in result["input_documents"]:
    print()
    print("-" * 100)
    print(d.metadata)
    print(d.page_content)

# COMMAND ----------

answer_question("Do you allow tenants to view your SAS70 Type II/SSAE 16 SOC2/ISAE3402 or similar third party audit reports?")

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator

qa_chain_b = sc.broadcast(qa_chain)

@pandas_udf('answer string, sources array<string>')
def answer_question_udf(question_sets: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
  db_udf = Chroma(collection_name="sec_docs", embedding_function=hf_embed, persist_directory=db_persist_path)
  for questions in question_sets:
    responses = []
    for question in questions:
      result = do_answer_question(question, db_udf, qa_chain_b.value)
      responses.append({"answer": result["output_text"], "sources": [d.metadata["source"] for d in result["input_documents"]]}) 
    yield pd.DataFrame(responses)

# COMMAND ----------

new_questions = spark.read.option("header", True).csv("/tmp/sean.owen@databricks.com/questionnaire.csv").\
  select("Question").toDF("question").\
  repartition(32)
display(new_questions)

# COMMAND ----------

# MAGIC %sh rm -r /dbfs/tmp/sean.owen@databricks.com/sec_rfp_langchain

# COMMAND ----------

write_path = "/tmp/sean.owen@databricks.com/sec_rfp_langchain"
response_df = new_questions.select(col("question"), answer_question_udf("question").alias("response")).select("question", "response.*")
response_df.write.format("delta").mode("overwrite").save(write_path)
display(spark.read.format("delta").load(write_path))

# COMMAND ----------


