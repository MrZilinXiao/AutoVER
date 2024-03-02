import os
from collections import defaultdict
import json

input_eval_file = "trie_bins/gpt4v-query-oven-0.5_exactmatch.jsonl"

counter = defaultdict(int)

with open(input_eval_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        try:
            split_name = data['data_split']
            counter[f"{split_name}_total"] += 1
            counter[f'{split_name}_correct'] += data['exact_match']
        except KeyError:
            print(data)
            pass

print(counter)

for split_name in sorted(counter.keys()):
    if split_name.endswith('_correct'):
        print(f"{split_name}: acc is {counter[split_name] / counter[split_name[:-8] + '_total']}")