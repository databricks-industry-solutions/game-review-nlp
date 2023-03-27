# Databricks notebook source
# MAGIC %pip install -U langchain chromadb pypdf pycryptodome openai tiktoken

# COMMAND ----------

import os
# Nicole's personal key - slack to get key
os.environ["OPENAI_API_KEY"] = ""

# COMMAND ----------

from pyspark.sql.functions import concat, lit, col

sqrc_texts = spark.read.option("header", True).option("multiLine", True).csv("/tmp/sean.owen@databricks.com/sqrc_answers.csv").\
  fillna("").\
  select(concat(lit("SQRC "), col("id")), concat(col("question"), lit(" "), col("platform-aws-mt_short_answer"), lit(" "), col("platform-aws-mt_detailed_answer"))).\
  toDF("source", "text")
display(sqrc_texts)

# COMMAND ----------

# MAGIC %sh rm -r /dbfs/tmp/sean.owen@databricks.com/langchain/db-openai ; mkdir -p /dbfs/tmp/sean.owen@databricks.com/langchain/db-openai

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#all_texts = sig_texts.union(sqrc_texts)
# Only use SQRC as it's a superset?
all_texts = sqrc_texts
documents = [Document(page_content=r["text"], metadata={"source": r["source"]}) for r in all_texts.collect()]
embeddings = OpenAIEmbeddings()
db_persist_path = "/dbfs/tmp/sean.owen@databricks.com/langchain/db-openai"
db = Chroma.from_documents(collection_name="sec_docs", documents=documents, embedding=embeddings, persist_directory=db_persist_path)

# COMMAND ----------

db.similarity_search("dummy") # tickle it to persist metadata (?)
db.persist()

# COMMAND ----------

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2, max_tokens=-1)
qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

# COMMAND ----------

def do_answer_question(question, db_local, qa_chain_local):
  similar_docs = db_local.similarity_search(question, k=8)
  return qa_chain_local({"input_documents": similar_docs, "question": question})

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
  os.environ["OPENAI_API_KEY"] = "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"
  db_udf = Chroma(collection_name="sec_docs", embedding_function=OpenAIEmbeddings(), persist_directory=db_persist_path)
  for questions in question_sets:
    responses = []
    for question in questions:
      result = do_answer_question(question, db_udf, qa_chain_b.value)
      responses.append({"answer": result["output_text"], "sources": [d.metadata["source"] for d in result["input_documents"]]}) 
    yield pd.DataFrame(responses)

# COMMAND ----------

new_questions = spark.read.option("header", True).csv("/tmp/sean.owen@databricks.com/questionnaire.csv").\
  select("Question").toDF("question").\
  repartition(2)
display(new_questions)

# COMMAND ----------

# MAGIC %sh rm -r /dbfs/tmp/sean.owen@databricks.com/sec_rfp_openai

# COMMAND ----------

write_path = "/tmp/sean.owen@databricks.com/sec_rfp_openai"
response_df = new_questions.select(col("question"), answer_question_udf("question").alias("response")).select("question", "response.*")
response_df.write.format("delta").mode("overwrite").save(write_path)
display(spark.read.format("delta").load(write_path))

# COMMAND ----------


