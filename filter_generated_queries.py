import os
import pandas as pd

def filter_dataset(corpus_path, generated_path):
    df = pd.read_json(corpus_path, lines=True)
    df = df.dropna()
    unique_ids = df['_id'].unique().tolist()[:1000]
    #filter the df to only include the unique ids
    df = df[df['_id'].isin(unique_ids)]
    df.to_json(generated_path, orient='records', lines=True)

data_path = "generated-queries"
output_path = "filtered_generated_queries"

os.makedirs(output_path, exist_ok=True)

for dataset in os.listdir(data_path):
    if dataset == ".DS_Store":
        continue
    os.makedirs(f"{output_path}/{dataset}", exist_ok=True)

    #check if corpus.jsonl exists in the dataset
    if os.path.isdir(f"{data_path}/{dataset}") and not os.path.exists(f"{data_path}/{dataset}/train.jsonl"):
        subfolders = os.listdir(f"{data_path}/{dataset}")
        for subfolder in subfolders:
            if subfolder == ".DS_Store" or subfolder == "README.md" or subfolder == ".gitattributes" or subfolder == ".git":
                continue
            if os.path.exists(f"{data_path}/{dataset}/{subfolder}/train.jsonl"):
                corpus_path = f"{data_path}/{dataset}/{subfolder}/train.jsonl"
                generated_path = f"{output_path}/{dataset}/{subfolder}/train.jsonl"
                os.makedirs(f"{output_path}/{dataset}/{subfolder}", exist_ok=True)
                print(corpus_path)
                filter_dataset(corpus_path, generated_path)

    elif os.path.exists(f"{data_path}/{dataset}/train.jsonl"):
        corpus_path = f"{data_path}/{dataset}/train.jsonl"
        generated_path = f"{output_path}/{dataset}/train.jsonl"
        print(corpus_path)
        filter_dataset(corpus_path, generated_path)