# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import sys
sub_paths = [
    "./LLaVA/"
]
for sub_path in sub_paths:
    sys.path.insert(0, sub_path)  # hack to allow import from the sub directory

from LLaVA.llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from train_oven import train

if __name__ == "__main__":
    train()
