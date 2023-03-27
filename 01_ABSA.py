# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/game-review-nlp. 

# COMMAND ----------

# MAGIC %md
# MAGIC Topics for Discussion
# MAGIC * Ontology - better grouping similar terms (warring faction game design vs. faction) and concepts/categories. Fine-tune on more ABSA data or create our own game-specific data to fine-tune?
# MAGIC * How does a studio fine-tune for their specific game and is that needed?
# MAGIC * Best way to surface insights:
# MAGIC   * Dashboard that shows dip in sentiment for something. From there, you click and read through the reviews and get feel for why the dip in sentiment.
# MAGIC   * Why do I need to sift through the reviews? Why can't it just summarize and slack me?
# MAGIC     * Impact on mental health of game designers
# MAGIC * Phase 2: multi-lingual
# MAGIC 
# MAGIC Other models to experiment with
# MAGIC * PyABSA

# COMMAND ----------

# DBTITLE 1,Prepare source data - download reviews from Steam
# MAGIC %run ./util/prepare-data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download and test model
# MAGIC * InstructABSA

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined")
model = AutoModelForSeq2SeqLM.from_pretrained("kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined")

bos_instruction = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
    Positive example 1-
    input: I charge it at night and skip taking the cord with me because of the good battery life.
    output: battery life:positive, 
    Positive example 2-
    input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
    output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
    Negative example 1-
    input: Speaking of the browser, it too has problems.
    output: browser:negative
    Negative example 2-
    input: The keyboard is too slick.
    output: keyboard:negative
    Neutral example 1-
    input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
    output: battery:neutral
    Neutral example 2-
    input: Nightly my computer defrags itself and runs a virus scan.
    output: virus scan:neutral
    Now complete the following example-
    input: """
delim_instruct = ''
eos_instruct = ' \noutput:'
text = 'The cab ride was amazing but the service was pricey.'

tokenized_text = tokenizer(bos_instruction + text + delim_instruct + eos_instruct, return_tensors="pt")
output = model.generate(tokenized_text.input_ids)
print('Model output: ', tokenizer.decode(output[0], skip_special_tokens=True))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test Inference

# COMMAND ----------

text = "New World is a game with infinite potential, but is handled by lazy and incompetent management. Mired by issues since it's launch, it was never able to recover. For reasons beyond all understanding, Amazon thought it wise to push out numerous bugfix patches without ever creating a Public Test server. Sure, they eventually did make one, but it was far too late. PvP was destroyed by bugs and exploits, and thanks to the mass-exodus of frustrated PvPers, the warring faction game design fell apart"

tokenized_text = tokenizer(bos_instruction + text + delim_instruct + eos_instruct, return_tensors="pt")
output = model.generate(tokenized_text.input_ids)
print('Model output: ', tokenizer.decode(output[0], skip_special_tokens=True))

# COMMAND ----------

text = "I was one of those hyped people that started right when New World launched...back then, once past the queues, i had the time of my life in a mmorpg. The server start was amazing, many people looking for fun. Despite all that people said, I actually enjoyed the leveling, but I am kind of a grinder in every game so dont take my word for granted in that case :D I stopped playing after a couple of weeks like a lot of people because of all the bugs and exploits, especially the duping exploits gave me"

tokenized_text = tokenizer(bos_instruction + text + delim_instruct + eos_instruct, return_tensors="pt")
output = model.generate(tokenized_text.input_ids)
print('Model output: ', tokenizer.decode(output[0], skip_special_tokens=True))

# COMMAND ----------

text = "I haven't reached level 60 yet (I'm close, at 54), but I'm already feeling a bit burned out thanks to a myriad of weird design choices on AGS' part. For starters, there's no ingame restriction so that a faction can't control 90% of the land, thus disabling 90% of the endgame content (faction warfare). On the server I'm playing right now, my own faction owns all but two cities, and this is mostly due to a big clan from another game just showing up in numbers and creating like 8 clans that just"

tokenized_text = tokenizer(bos_instruction + text + delim_instruct + eos_instruct, return_tensors="pt")
output = model.generate(tokenized_text.input_ids)
print('Model output: ', tokenizer.decode(output[0], skip_special_tokens=True))

# COMMAND ----------


