from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
model_paths = ["full"]
data_paths = []
datasets = os.listdir("datasets")
for dataset in datasets:
    data_path = f"datasets/{dataset}"
    if not os.path.isdir(data_path):
        continue
    if dataset == "cqadupstack":
        subfolders = os.listdir(data_path)
        for subfolder in subfolders:
            if not os.path.isdir(f"{data_path}/{subfolder}"):
                continue
            data_paths.append(f"{data_path}/{subfolder}")
    else:
        data_paths.append(data_path)

for model_path in model_paths:
    os.makedirs(f"results/{model_path}", exist_ok=True)
    for data_path in data_paths:
        print(f"Processing {data_path}")           
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        sub_folder = "/".join(data_path.split("/")[1:])
        #### Load the SBERT model and retrieve using cosine-similarity
        model = DRES(models.SentenceBERT(f"{model_path}/{sub_folder}"), batch_size=16)
        retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
        results = retriever.retrieve(corpus, queries)

        #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        result_dict = {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision}
        os.makedirs(f"results/{model_path}/{sub_folder}", exist_ok=True)
        with open(f"results/{model_path}/{sub_folder}/results.json", "w") as f:
            json.dump(result_dict, f, indent=2)
            
