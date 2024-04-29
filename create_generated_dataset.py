import pandas as pd
import json

data_path = "fiqa-generated-queries/train.jsonl"

data = pd.read_json(data_path, lines=True)

json_files = []
print(len(data))
for i in range(len(data)):
    data_dict = {"_id": f"genQ{i+1}", "text": data.iloc[i]["query"], "metadata": {}}
    json_files.append(data_dict)

# save json_files to a jsonl file
output_path = "generated/datasets/fiqa/qgen-queries.jsonl"
with open(output_path, "w") as f:
    for line in json_files:
        f.write(json.dumps(line) + "\n")

query_ids = []
corpus_ids = []
scores = []
for i in range(len(data)):
    query_ids.append(f"genQ{i+1}")
    corpus_ids.append(data.iloc[i]["_id"])
    scores.append(1)

df = pd.DataFrame({"query-id": query_ids, "corpus-id": corpus_ids, "score": scores})

output_path = "generated/datasets/fiqa/qgen-qrels/train.tsv"

df.to_csv(output_path, sep="\t", index=False)

"""import os

data_path = "cqadupstack-generated-queries"

for dataset in os.listdir(data_path):
    # check if the file is a directory
    if not os.path.isdir(f"{data_path}/{dataset}") or dataset == "README.md" or dataset == ".gitignore" or dataset == ".DS_Store" or dataset == ".git":
        continue

    if dataset in os.listdir("generated/datasets/cqadupstack"):
        continue

    print(f"Processing {dataset}")
    
    data = pd.read_json(f"{data_path}/{dataset}/train.jsonl", lines=True)

    json_files = []
    print(len(data))
    for i in range(len(data)):
        data_dict = {"_id": f"genQ{i+1}", "text": data.iloc[i]["query"], "metadata": {}}
        json_files.append(data_dict)

    os.makedirs(f"generated/datasets/cqadupstack/{dataset}", exist_ok=True)
    # save json_files to a jsonl file
    output_path = f"generated/datasets/cqadupstack/{dataset}/qgen-queries.jsonl"
    with open(output_path, "w") as f:
        for line in json_files:
            f.write(json.dumps(line) + "\n")

    query_ids = []
    corpus_ids = []
    scores = []
    for i in range(len(data)):
        query_ids.append(f"genQ{i+1}")
        corpus_ids.append(data.iloc[i]["_id"])
        scores.append(1)

    df = pd.DataFrame({"query-id": query_ids, "corpus-id": corpus_ids, "score": scores})

    os.makedirs(f"generated/datasets/cqadupstack/{dataset}/qgen-qrels", exist_ok=True)
    output_path = f"generated/datasets/cqadupstack/{dataset}/qgen-qrels/train.tsv"

    df.to_csv(output_path, sep="\t", index=False)"""
    