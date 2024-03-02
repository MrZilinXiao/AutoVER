# 100 requests per day: so we have to batching the queries
# 100 RPD; 20000 TPM; each query takes 200 tokens on average; so a batch has 100 queries; 1 day can handle 100 batches = 10000 queries
import os
import backoff
import json
import base64
import requests
from tqdm import tqdm

api_key = "XXXX"

# global shared variables
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_img_path(image_id: str):
    shard_id = image_id.split('_')[1][:2]
    for ext in ('.jpg', '.JPEG'):
        img_path = os.path.join("/vc_data/users/v-zilinxiao/dataset/entity_oven/", f"{shard_id}", f"{image_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    raise RuntimeError(f"Image {image_id} not found in {img_path}")

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
def submit_oven_gpt4v(image_id, query_item):
    image_path = find_img_path(image_id)
    base64_image = encode_image(image_path)
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query_item['question']
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}", 
                    "detail": "low"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
    query_item.update({'gpt4v_response': response.json()})
    if 'error' in query_item['gpt4v_response']:
        print(f"Error: {query_item['gpt4v_response']}")
        raise RuntimeError()
    return query_item

# open 0.1 split and ready to submit
with open('../ms_jsonl_dataset/oven_entity_val_0.1.jsonl', 'r') as f:
    oven_val = [json.loads(line) for line in f.readlines()]

# gather oven data_id; skip generated queries
with open('../trie_bins/gpt4v-eval-oven-0.1.jsonl', 'r') as f:
    generated_data_ids = [json.loads(line)['data_id'] for line in f.readlines()]

with open('../trie_bins/gpt4v-eval-oven-0.1.jsonl', 'a') as f:
    for i, query_item in enumerate(tqdm(oven_val)):
        if query_item['data_id'] in generated_data_ids:
            continue
        image_id = query_item['image_id']
        query_item = submit_oven_gpt4v(image_id, query_item)
        f.write(json.dumps(query_item) + '\n')
        f.flush()
        if i % 100 == 0:
            print(f"Finished {i} images, {query_item}")