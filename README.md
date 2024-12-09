# How-to Guide
## Using vLLM + Langchain + LangServe + Milvus
### Provision the infrastructure
```shell
docker network create mlek3
cd deploy/local
docker compose -f milvus-docker-compose.yaml -f vllm-docker-compose.yaml up -d
```
### Ingest data into Milvus vector database
#### 1. Prepare data

In this example, we are going to use CNN daily news for our knowledge base.

Below are the steps I took to generate the file `cnn_dailymail/3.0.0/train-00000-of-00003.parquet`. You can re-run the following commands to re-generate or simply skip them.

```shell
git clone https://huggingface.co/datasets/abisee/cnn_dailymail
cd cnn_dailymail
git lfs install
git lfs pull
rm -rf 1.0.0/ 2.0.0/
cd 3.0.0/
rm train-00001-of-00003.parquet train-00002-of-00003.parquet test*.parquet validation*.parquet
```

#### 2. Ingest data into the vector database

Please create a new Conda environment with `Python=3.10.14` and open the following Jupyter Lab
```shell
jupyter lab
```
and execute the notebook `notebooks/rag.ipynb`.

Please read the notebook carefully, it will also guide you how to make your first LLM prediction with vLLM.

### Enjoy your application

Open Swagger UI via `http://localhost:8001/docs`, and make similar actions as follows

![Example Output on FastAPI](./imgs/Example%20Output%20on%20FastAPI%201.png)

![Example Output on FastAPI](./imgs/Example%20Output%20on%20FastAPI%202.png)


## Triton Server with vLLM backend (optiona)

This section is optional since it is too new and poorly supported by other frameworks, such as Langchain, and has a small community. However, deploying a model this way is very simple.

### 1. Prepare the model
Create a new folder `model_repository` with the following structure:
```shell
.
├── 1
│   └── model.json
└── config.pbtxt
```

### 2. Deploy the infrastructure

```shell
cd deploy/local
docker compose -f triton-vllm-docker-compose.yaml up -d
```
Note:
- You can freely replace `24.05` by other versions.
- If you want to change some configuration options, for example, update vLLM backend version, please [build a custom container](https://github.com/triton-inference-server/vllm_backend?tab=readme-ov-file#option-2-build-a-custom-container-from-source) instead of using the default. Sometimes, it can also save you a lot of computing resources.

### 3. Enjoy your result

```shell
curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "Tell me about Hanoi", "parameters": {"stream": false, "temperature": 0}}'
```