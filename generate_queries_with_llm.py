import os
import json
import pandas as pd
from openai import OpenAI
from decouple import config

client = OpenAI(api_key=config('API_KEY'))

def process_corpus(corpus_path, output_path):
    with open(corpus_path, "r") as f:
        corpus = [json.loads(line) for line in f]
    
    out_file = open(output_path, "w")
    
    df = pd.DataFrame(corpus)
    
    unique_ids = df['_id'].unique().tolist()
    for _id in unique_ids:
        df_id = df[df['_id'] == _id]
        text = df_id['text'].tolist()[0]
        title = df_id['title'].tolist()[0]
        number_of_queries = len(df_id)
        queries = generate_queries(text, number_of_queries)
        processed_queries = process_queries(queries)
        for query in processed_queries:
            out_file.write(json.dumps({"_id": _id, "title": title, "text":text, "query": query}) + "\n")
    out_file.close()

def generate_queries(text, number_of_queries=3):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Given a block of text, generate three queries based on the content of the text. These queries will be used for pseudo-labeling of a retrieval model's training data using sentence-BERT. The goal is to create relevant and diverse queries that capture the essence of the text and can be used to retrieve similar information."},
            {"role": "user", "content": f"Please provide three queries based on the following text. These queries will be used to improve the performance of a retrieval model through pseudo-labeling with sentence-BERT. The text is as follows: {text} Your Task: Generate {number_of_queries} queries that effectively summarize the content of the text and can be used to retrieve similar information. Consider the key concepts, entities, and context provided in the text to craft diverse and relevant queries. Each query should be distinct and capture different aspects of the text. Output Format:Generated queries should be separated by new line, there shouldn't be any output other than generated queries."}
        ]
    )
    queries = completion.choices[0].message.content.split("\n")
    return queries

def process_queries(queries):
    processed_queries = []
    for query in queries:
        query = query[3:]
        processed_queries.append(query)
    return processed_queries

data_path = "generated-queries"
output_path = "llm_generated_queries"

os.makedirs(output_path, exist_ok=True)

for dataset in os.listdir(data_path):
    print(f"Processing {dataset}")
    os.makedirs(f"{output_path}/{dataset}", exist_ok=True)

    #check if corpus.jsonl exists in the dataset
    if os.path.isdir(f"{data_path}/{dataset}") and not os.path.exists(f"{data_path}/{dataset}/train.jsonl"):
        subfolders = os.listdir(f"{data_path}/{dataset}")
        for subfolder in subfolders:
            print(f"Processing {subfolder}")
            if os.path.exists(f"{data_path}/{dataset}/{subfolder}/train.jsonl"):
                corpus_path = f"{data_path}/{dataset}/{subfolder}/train.jsonl"
                output_path = f"{output_path}/{dataset}/{subfolder}/train.jsonl"
                process_corpus(corpus_path, output_path)

    elif os.path.exists(f"{data_path}/{dataset}/train.jsonl"):
        corpus_path = f"{data_path}/{dataset}/train.jsonl"
        output_path = f"{output_path}/{dataset}/train.jsonl"
        process_corpus(corpus_path, output_path)
        
