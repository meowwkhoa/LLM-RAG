from fastapi import FastAPI
from langserve import add_routes
from chain import qa_chain
from constants import *
import pandas as pd
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

from langchain.agents import tool
from langchain_core.runnables import RunnableLambda

app = FastAPI(title="RAG Pipeline API")

# Add QA chain to our app
# add_routes(app, qa_chain)

# Connect to Milvus
milvus_client = MilvusClient(uri=MILVUS_HOST)

# Create a collection to store vectors and metadata
schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(
    schema=schema,
    field_name="id",
    datatype=DataType.VARCHAR,
    is_primary=True,
    max_length=10000,
)
schema.add_field(
    schema=schema, field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM
)
schema.add_field(
    schema=schema, field_name="text", datatype=DataType.VARCHAR, max_length=10000
)

# Drop the old collection and create a new one
if milvus_client.has_collection(MILVUS_COLLECTION_NAME):
    milvus_client.drop_collection(MILVUS_COLLECTION_NAME)

milvus_client.create_collection(collection_name=MILVUS_COLLECTION_NAME, schema=schema)

# Define the embedding model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# @tool
# def ingest_parquet():
#     # Read the Parquet file
#     df = pd.read_parquet("/home/khoa/MLE-k3/Supplementary/llm-source-code/cnn_dailymail/3.0.0/train-00000-of-00003.parquet")
#     df = df[["id", "article"]]
#     articles = df.iloc[:NUM_ARTICLES].to_dict("records")

#     # Loop over articles and create embedding vectors
#     data = []
#     for article in articles:
#         data.append(
#             {
#                 "id": article["id"],
#                 "vector": model.encode(article["article"]),
#                 "text": article["article"],
#             }
#         )
#     milvus_client.insert(collection_name=MILVUS_COLLECTION_NAME, data=data)
        
#     return {"status": "success"}

# @tool
# def chat(query: str):
#     result = qa_chain.invoke({"query": query})
#     return result

@app.post("/ingest_parquet")
async def ingest_parquet():
    # Read the Parquet file
    df = pd.read_parquet("/home/khoa/MLE-k3/Supplementary/llm-source-code/cnn_dailymail/3.0.0/train-00000-of-00003.parquet")
    df = df[["id", "article"]]
    articles = df.iloc[:NUM_ARTICLES].to_dict("records")

    # Loop over articles and create embedding vectors
    data = []
    for article in articles:
        data.append(
            {
                "id": article["id"],
                "vector": model.encode(article["article"]),
                "text": article["article"],
            }
        )
    milvus_client.insert(collection_name=MILVUS_COLLECTION_NAME, data=data)
        
    return {"status": "success"}

@app.post("/chat")
async def chat(query: str):
    answer = qa_chain.invoke({"query": query})
    return {"question": query, "answer": answer}

# ingest_parquet_runnable = RunnableLambda(ingest_parquet)
# chat_runnable = RunnableLambda(chat)

# add_routes(app, ingest_parquet_runnable, path="/ingest_parquet", methods=["POST"])

# add_routes(app, chat_runnable, path="/chat", methods=["POST"])