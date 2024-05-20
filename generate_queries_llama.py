from langchain_community.llms import Ollama
from langchain import PromptTemplate # Added
import os
import json
import pandas as pd

llm = Ollama(model="llama3", stop=["<|eot_id|>"]) # Added stop token

def get_model_response(user_prompt, system_prompt):
    # NOTE: No f string and no whitespace in curly braces
    template = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

    # Added prompt template
    prompt = PromptTemplate(
        input_variables=["system_prompt", "user_prompt"],
        template=template
    )
    
    # Modified invoking the model
    response = llm(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))
    
    return response

def prepare_system_and_user_prompts(text, number_of_queries=3):
    system_prompt = "Given a block of text, generate three queries based on the content of the text. These queries will be used for pseudo-labeling of a retrieval model's training data using sentence-BERT. The goal is to create relevant and diverse queries that capture the essence of the text and can be used to retrieve similar information."

    user_prompt = f"Please provide three queries based on the following text. These queries will be used to improve the performance of a retrieval model through pseudo-labeling with sentence-BERT. The text is as follows: {text} Your Task: Generate {number_of_queries} queries that effectively summarize the content of the text and can be used to retrieve similar information. Consider the key concepts, entities, and context provided in the text to craft diverse and relevant queries. Each query should be distinct and capture different aspects of the text. Output Format:Generated queries should be separated by new line, there shouldn't be any output other than generated queries."

    return system_prompt, user_prompt

def process_corpus(corpus_path, output_path):
    #Processing 132512
    with open(corpus_path, "r") as f:
        corpus = [json.loads(line) for line in f]
    
    out_file = open(output_path, "w")
    
    df = pd.DataFrame(corpus)
    
    unique_ids = df['_id'].unique().tolist()
    for _id in unique_ids:
        print(f"Processing {_id}")
        df_id = df[df['_id'] == _id]
        text = df_id['text'].tolist()[0]
        title = df_id['title'].tolist()[0]
        number_of_queries = len(df_id)
        system_prompt, user_prompt = prepare_system_and_user_prompts(text, number_of_queries)
        response = get_model_response(user_prompt, system_prompt)
        out_file.write(json.dumps({"_id": _id, "title": title, "text":text, "response": response}) + "\n")
    out_file.close()

data_path = "filtered_generated_queries"
output_path = "llm_generated_queries"

os.makedirs(output_path, exist_ok=True)

for dataset in os.listdir(data_path):
    if dataset == ".DS_Store":
        continue
    print(f"Processing {dataset}")
    os.makedirs(f"{output_path}/{dataset}", exist_ok=True)

    #check if corpus.jsonl exists in the dataset
    if os.path.isdir(f"{data_path}/{dataset}") and not os.path.exists(f"{data_path}/{dataset}/train.jsonl"):
        subfolders = os.listdir(f"{data_path}/{dataset}")
        for subfolder in subfolders:
            if subfolder == ".DS_Store" or subfolder == "README.md" or subfolder == ".gitattributes" or subfolder == ".git":
                continue
            print(f"Processing {subfolder}")
            os.makedirs(f"{output_path}/{dataset}/{subfolder}", exist_ok=True)
            if os.path.exists(f"{data_path}/{dataset}/{subfolder}/train.jsonl"):
                corpus_path = f"{data_path}/{dataset}/{subfolder}/train.jsonl"
                generated_path = f"{output_path}/{dataset}/{subfolder}/train.jsonl"
                process_corpus(corpus_path, generated_path)

    elif os.path.exists(f"{data_path}/{dataset}/train.jsonl"):
        corpus_path = f"{data_path}/{dataset}/train.jsonl"
        generated_path = f"{output_path}/{dataset}/train.jsonl"
        process_corpus(corpus_path, generated_path)