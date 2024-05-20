import pandas as pd

data = pd.read_csv('generated/datasets/trec-covid/qgen-qrels/train.tsv', sep='\t')

corpus = pd.read_json('generated/datasets/trec-covid/corpus.jsonl', lines=True)

not_found = []
for corpus_id in data['corpus-id'].unique().tolist():
    if corpus_id not in corpus['_id'].values:
        print(f"Corpus ID {corpus_id} not found in corpus.jsonl")
        not_found.append(corpus_id)
        continue

print(f"Total not found: {len(not_found)}")
print(f"Total found: {len(data['corpus-id'].unique().tolist()) - len(not_found)}")
    #out_file.write(json.dumps({"_id": corpus_id, "title": title, "text":text, "response": response}) + "\n")