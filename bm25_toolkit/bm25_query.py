import argparse
import json
# import time
import pickle
from multiprocessing import Pool
from transformers import AutoTokenizer
# from rank_bm25 import BM25Okapi
from tqdm import tqdm

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

# def run_query(query):
#     tokenized_query = tokenizer.tokenize(query["prediction"])
#     results = bm25.get_top_n(tokenized_query, corpus, n=5)
#     return {"data_id": query["data_id"], "prediction": query["prediction"], "bm25": results}

def run_gpt_query(query):
    # query sample: {"data_id": "oven_query_val_00000018", "image_id": "oven_04925757", 
    # "question": "what kind of food is this kid eating?", "entity_id": "Q13317", 
    # "entity_text": "Yogurt", "data_split": "query_val_seen", 
    # "gpt4v_response": {"id": "chatcmpl-8Q7epBIjpiMYDeM8qbD2cp7rUVQDh", 
    # "object": "chat.completion", "created": 1701236655, "model": 
    # "gpt-4-1106-vision-preview", "usage": {"prompt_tokens": 101, 
    # "completion_tokens": 68, "total_tokens": 169}, "choices": 
    # [{"message": {"role": "assistant", "content": "The child in 
    # the image appears to be eating a glazed donut from a bowl. 
    # There is also a package of Hostess Donettes, which are mini donuts,
    #  shown in the hands of an adult next to him. This suggests that 
    # the child might be enjoying some sweet, bakery-style snacks, 
    # likely donettes from the package displayed."}, "finish_details": 
    # {"type": "stop", "stop": "<|fim_suffix|>"}, "index": 0}]}}
    gpt4_response = query.pop("gpt4v_response")
    if 'error' in gpt4_response:  # error due to content_policy_violation results in empty response
        return {"data_id": query["data_id"], "prediction": "content_policy_violation error", "bm25": []}
    prediction = gpt4_response['choices'][0]['message']['content']
    tokenized_query = tokenizer.tokenize(prediction)
    results = bm25.get_top_n(tokenized_query, corpus, n=5)
    print(f"Query: {prediction}; results: {results}; answer: {query['entity_text']}")
    return {"data_id": query["data_id"], "prediction": prediction, "bm25": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, default="trie_bins/gpt4v-eval-oven-0.1.jsonl")
    # parser.add_argument("--output_file", type=str, default="trie_bins/gpt4v-eval-oven-0.1_bm25.jsonl")
    parser.add_argument("--input_file", type=str, default="trie_bins/gpt4v-query-oven-0.5.jsonl")
    parser.add_argument("--output_file", type=str, default="trie_bins/gpt4v-query-oven-0.5_bm25.jsonl")
    args = parser.parse_args()

    corpus = load_json("trie_bins/Wiki6M_ver_1_0_title_only.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    corpus = [
        [d["wikipedia_title"], d["wikidata_id"]] for d in corpus
    ]

    print("Loading bm25 pickle...")
    with open("wikipedia6m_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    input_query = load_json(args.input_file)

    print("Running bm25 query...")
    with Pool() as p:
        max_ = len(input_query)
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(run_gpt_query, input_query))):
                pbar.update()
        output = list(p.imap_unordered(run_gpt_query, input_query))

    with open(args.output_file, "w", encoding="utf-8") as f:
        for d in output:
            f.write(json.dumps(d) + "\n")