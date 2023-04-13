# Databricks notebook source
# MAGIC %md
# MAGIC Use Databricks Runtime 12.2 ML or higher. 
# MAGIC GPU cluster is not necessary. Use a high-mem type, and set `spark.task.cpus` to 8.
# MAGIC 
# MAGIC Please run the `RUNME` notebook to generate a cluster that satisfies these requirement.

# COMMAND ----------

# MAGIC %pip install -U transformers langchain chromadb pypdf pycryptodome accelerate

# COMMAND ----------

# MAGIC %md
# MAGIC Read RFP responses from a CSV file. The desired inputs are 'documents', or chunks of text containing question and answer. This is retrieved along with the source for reference.

# COMMAND ----------

dbutils.widgets.dropdown("reuse_existing_vector_store", "True" ,["True", "False"])
reuse_existing_vector_store = dbutils.widgets.get("reuse_existing_vector_store") == "True"

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

# MAGIC %sql use cme_corey_abshire_dev --REMOVE

# COMMAND ----------

# select new world reviews longer than X characters - We want to focus on longer reviews because they tend to contain more in-depth discussions to support QA

review_texts = spark.sql("""select recommendationid, timestamp_created, review from steam_appreviews_bronze where appid = '1063730' and len(review) > 1000""") 

# display(review_texts)

# COMMAND ----------

# MAGIC %md 

# COMMAND ----------

# MAGIC %md As we have seen in the preview above, some reviews can go into great depth and be quite long. We may need to split the reviews into chunks and do so in a smart way to keep semantically related parts of texts together. [Doc](https://python.langchain.com/en/latest/modules/indexes/text_splitters/getting_started.html)

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 0,
    length_function = len,
)
collected_review = review_texts.collect()
collected_review_text = [r["review"] for r in collected_review]
collected_review_metadatas = [{"source": r["recommendationid"]} for r in collected_review]

documents = text_splitter.create_documents(collected_review_text, collected_review_metadatas)

# COMMAND ----------

# DBTITLE 1,Initialize the location for our vector store
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

db_persist_path_chroma = f"{cloud_storage_path}langchain/chroma/db" 
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
chroma_exists = os.path.isfile(f"/dbfs{db_persist_path_chroma}/chroma-embeddings.parquet")

if reuse_existing_vector_store and chroma_exists:
  db = Chroma(collection_name="new_world_reviews", persist_directory=f"/dbfs{db_persist_path_chroma}", embedding_function=hf_embed)
else:
  dbutils.fs.rm(db_persist_path_chroma, True)
  dbutils.fs.mkdirs(db_persist_path_chroma)
  db = Chroma.from_documents(collection_name="new_world_reviews", documents=documents, embedding=hf_embed, persist_directory=f"/dbfs{db_persist_path_chroma}")
  db.similarity_search("dummy") # tickle it to persist metadata (?)
  db.persist()

# COMMAND ----------

# MAGIC %md
# MAGIC Build the actual chain. The chain will contain:
# MAGIC - A reference to the document DB, so that new queries can be embedded and matched to text
# MAGIC - A prompt template controlling how the context docs and question are fed to a language model
# MAGIC - A language model

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

def build_qa_chain():
  model_name = "databricks/dolly-v1-6b"
  tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
  model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

  template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  Use only information in the following paragraphs to answer the question at the end. Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.

  {context}

  {question}

  ### Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

  end_key_token_id = tokenizer.encode("### End")[0]

  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, \
    pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id, \
    do_sample=False, max_new_tokens=256, num_beams=2, num_beam_groups=2)

  # Increase max_new_tokens for a longer response
  # Other settings might give better results! Play around
  #pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, \
  #  pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id, \
  #  do_sample=True, max_new_tokens=128, top_p=0.95, top_k=50)

  hf_pipe = HuggingFacePipeline(pipeline=pipe)
  # Set verbose=True to see the full prompt:
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC Aspects
# MAGIC 
# MAGIC * Player versus player (pvp) 
# MAGIC * Crafting 
# MAGIC * Endgame 
# MAGIC * Dungeons 
# MAGIC * Cheating 
# MAGIC * Questing 
# MAGIC 
# MAGIC 
# MAGIC What are the negative points related to player versus player (pvp) and factions?
# MAGIC What are the positive points related to player versus player (pvp) and factions?
# MAGIC How is the endgame experience?
# MAGIC Are the dungeons fun?
# MAGIC Is cheating happening?
# MAGIC How is the leveling or questing experience?

# COMMAND ----------

qa_chain = build_qa_chain()

def answer_question(question, k=2):
  similar_docs = db.similarity_search(question, k=k)
  result = qa_chain({"input_documents": similar_docs, "question": question})
  print(result["output_text"])
  # Optional: print the text of the sources that were used
  for d in result["input_documents"]:
    print("-" * 100)
    print(d.metadata)
    print(d.page_content)

  print("=" * 100)

# COMMAND ----------

question_list = ["What are the negative points related to player versus player (pvp) and factions?",
"What are the positive points related to player versus player (pvp) and factions?",
"How is the endgame experience?",
"Are the dungeons fun?",
"Is cheating happening?",
"How is the leveling or questing experience?"]

# COMMAND ----------

for q in question_list:
  print(f"\n== QUESTION: {q}\n")
  response = answer_question(q, k=4)
  print(f"== RESPONSE: {response}")

# COMMAND ----------

# from pyspark.sql.functions import pandas_udf
# import pandas as pd
# from typing import Iterator

# qa_chain_b = sc.broadcast(qa_chain)

# @pandas_udf('answer string, sources array<string>')
# def answer_question_udf(question_sets: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
#   db_udf = Chroma(collection_name="sec_docs", embedding_function=hf_embed, persist_directory=db_persist_path)
#   for questions in question_sets:
#     responses = []
#     for question in questions:
#       result = do_answer_question(question, db_udf, qa_chain_b.value)
#       responses.append({"answer": result["output_text"], "sources": [d.metadata["source"] for d in result["input_documents"]]}) 
#     yield pd.DataFrame(responses)

# COMMAND ----------

# new_questions = spark.read.option("header", True).csv("/tmp/sean.owen@databricks.com/questionnaire.csv").\
#   select("Question").toDF("question").\
#   repartition(32)
# display(new_questions)

# COMMAND ----------

# write_path = f"{cloud_storage_path}langchain/summary"
# response_df = new_questions.select(col("question"), answer_question_udf("question").alias("response")).select("question", "response.*")
# response_df.write.format("delta").mode("overwrite").save(write_path)
# display(spark.read.format("delta").load(write_path))

# COMMAND ----------


