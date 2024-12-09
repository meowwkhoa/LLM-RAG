from langchain.chains import RetrievalQA
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_community.vectorstores.milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from constants import *

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=VLLM_HOST,
    model_name=LLM_NAME,
    max_tokens=MAX_TOKENS,
    top_p=TOP_P,
    temperature=TEMPERATURE,
    streaming=True,
    verbose=False,
)

model_kwargs = {"trust_remote_code": True, "device": "cpu"}

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs, show_progress=False
)

store = Milvus(
    embedding_function=embeddings,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    collection_name=MILVUS_COLLECTION_NAME,
    text_field="text",
    drop_old=False,
)
retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)
