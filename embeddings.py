#!/usr/bin/env python
# coding: utf-8

# In[3]:


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

#OPENAI_API_KEY = os.getenv("openai_api_key")

# Configure logging
logging.basicConfig(
    filename="./scraping_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def log_update_date(message):
    """Log a custom message with the current date."""
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{message} - Date: {current_date}")


# Reading the CSV files into pandas DataFrames
df_dil = pd.read_csv('dil_scraped_data.csv')
df_isss = pd.read_csv('isss_scraped_data.csv')
df_csee = pd.read_csv('research_data.csv')

data = pd.concat([df_dil, df_isss], ignore_index=True)
# Ensure the 'Text' column contains strings and handle missing values
data["Text"] = data["Text"].fillna("").astype(str)
logging.info("Embeddings Creation")
# Initialize text splitter and embedding model
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize Chroma with persistence
persist_directory = "chroma_store"
chroma_store = Chroma(
    collection_name="retriever_bot",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

# Process each row in the dataset
for idx, row in data.iterrows():
    # Validate metadata fields
    if not (row["Section"] and row["Link"] and row["Title"]):
        print(f"Skipping row {idx} due to missing metadata.")
        continue

    # Split the text into chunks and filter out empty chunks
    chunks = [chunk for chunk in text_splitter.split_text(row["Text"]) if chunk.strip()]
    if not chunks:
        print(f"No valid chunks for row {idx}, skipping.")
        continue

    # Prepare metadata
    metadata = {
        "section": row["Section"],
        "link": row["Link"],
        "title": row["Title"]
    }

    # Add chunks to Chroma
    try:
        chroma_store.add_texts(texts=chunks, metadatas=[metadata] * len(chunks))
    except Exception as e:
        print(f"Error adding texts for row {idx}: {e}")

logging.info("Data added to Chroma and saved to disk.")

# Initialize another persistent directory
persist_directory_research = "chroma_store1"

# Initialize Chroma with persistence for research data
chroma_store1 = Chroma(
    collection_name="research_info",
    embedding_function=embedding_model,
    persist_directory=persist_directory_research
)

# Process each row in the dataset for research info
for idx, row in df_csee.iterrows():
    # Prepare metadata with Section, Link, and Title
    metadata = {
        "section": row["Section"],
        "link": row["Link"],
        "title": row["Title"]
    }

    # Add full text with metadata to Chroma without chunking
    chroma_store1.add_texts(
        texts=[row["Text"]],
        metadatas=[metadata]
    )

logging.info("Research data added to Chroma and saved to disk.")
log_update_date('Completed Successfully- Data added to chroma db')


# In[ ]:




