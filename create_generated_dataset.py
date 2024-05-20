import pandas as pd
import json
import os

def process_corpus(data_path, output_path):
    data = pd.read_json(data_path, lines=True)
    json_files = []
    print(len(data))
    for i in range(len(data)):
        data_dict = {"_id": f"genQ{i+1}", "text": data.iloc[i]["query"], "metadata": {}}
        json_files.append(data_dict)

    # save json_files to a jsonl file
    with open(f"{output_path}/qgen-queries.jsonl", "w") as f:
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
    os.makedirs(f"{output_path}/qgen-qrels", exist_ok=True)
    df.to_csv(f"{output_path}/qgen-qrels/train.tsv", sep="\t", index=False)

data_path = "filtered_generated_queries"
output_path = "generated"

os.makedirs(output_path, exist_ok=True)

for dataset in os.listdir(data_path):
    if dataset == ".DS_Store":
        continue
    print(f"Processing {dataset}")
    out_path = f"{output_path}/datasets/{dataset.replace('-generated-queries', '')}"
    os.makedirs(out_path, exist_ok=True)

    #check if corpus.jsonl exists in the dataset
    if os.path.isdir(f"{data_path}/{dataset}") and not os.path.exists(f"{data_path}/{dataset}/train.jsonl"):
        subfolders = os.listdir(f"{data_path}/{dataset}")
        for subfolder in subfolders:
            if subfolder == ".DS_Store" or subfolder == "README.md" or subfolder == ".gitattributes" or subfolder == ".git":
                continue
            print(f"Processing {subfolder}")
            out_path_sub = f"{out_path}/{subfolder}"
            os.makedirs(out_path_sub, exist_ok=True)
            if os.path.exists(f"{data_path}/{dataset}/{subfolder}/train.jsonl"):
                corpus_path = f"{data_path}/{dataset}/{subfolder}/train.jsonl"
                process_corpus(corpus_path, out_path_sub)

    elif os.path.exists(f"{data_path}/{dataset}/train.jsonl"):
        corpus_path = f"{data_path}/{dataset}/train.jsonl"
        process_corpus(corpus_path, out_path)