import gpl
import os

data_paths = []
for folder in os.listdir('datasets'):
    if os.path.isdir(os.path.join('datasets', folder)):
        if folder == 'cqadupstack':
            for subfolder in os.listdir(os.path.join('datasets', folder)):
                if os.path.isdir(os.path.join('datasets', folder, subfolder)):
                    data_paths.append(os.path.join('datasets', folder, subfolder))
        else:
            data_paths.append(os.path.join('datasets', folder))
        
for dataset in data_paths:
    print(f"Training on {dataset}")

    gpl.train(
        path_to_generated_data=f"generated/{dataset}",
        base_ckpt="distilbert-base-uncased",  
        gpl_score_function="dot",
        batch_size_gpl=32,
        batch_size_generation=8,
        gpl_steps=10,
        new_size=-1,
        queries_per_passage=-1,
        output_dir=f"output/{dataset}",
        evaluation_data=f"./{dataset}",
        evaluation_output=f"evaluation/{dataset}",
        generator="BeIR/query-gen-msmarco-t5-base-v1",
        retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        retriever_score_functions=["cos_sim", "cos_sim"],
        cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
        qgen_prefix="qgen",
        do_evaluation=True,
    )