# Grounding Language Models for Visual Entity Recognition

Official Implementation of "Grounding Language Models for Visual Entity Recognition". Code, checkpoints and documents are being updated.

Authors: Zilin Xiao, Ming Gong, Paola Cascante-Bonilla, Xingyao Zhang, Jie Wu, Vicente Ordonez

[\[Paper\]](https://arxiv.org/abs/2402.18695)

## Dataset Preparation
TBD

## Training

Training an AutoVER-7B takes 2.5 days on 32 V100 GPUs approximately.

```bash
deepspeed --num_gpus 8 --num_nodes 4 train_oven_ret.py \
    --deepspeed ./deepspeed_configs/zero3.json \
    --model_name_or_path ./vicuna-7b-v1.5-32064-vocab \
    --version v1 \
    --train_jsonl_path ./ms_jsonl_dataset/oven_entity_train_0.5.jsonl \
    --query_jsonl_path ./ms_jsonl_dataset/oven_query_train.jsonl \
    --train_img_shards_path $DATASET_PATH/entity_oven/ \
    --qid_to_summary_path ./trie_bins/wiki_qid_to_summary_indata.json \
    --entity_img_path $DATASET_PATH/wikipedia_images_full \
    --retrieval True \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --fp16 \
    --output_dir ./checkpoints/llava-v1.5-7b-ms-ret-0.5-trm-feedback2-fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name llava-7b-ms-ret-0.5-trm-feedback2-fp16 \
    --log_on_each_node False \
    --one_class_per_batch \
    --gather_contrastive True \
    --entity_encoder_fusion trm \
    --ret_to_entity_only True \
    --unfreeze_entity_clip False
```

## Evaluation

### AutoVER Evaluation

### GPT4-V Evaluation

Submit queries to GPT4-V endpoints with `baselines/submit_gpt4v.py`.

We have provided raw response from GPT4-V in `baselines/gpt4v-eval-oven-0.1.jsonl` and `baselines/gpt4v-query-oven-0.5.jsonl`.

### Other Genrative Baseline Evaluation

## Acknowledgement

Our implementation partially relies on the following repositories:

1. Base model architecture from [LLaVA](https://github.com/haotian-liu/LLaVA).
2. Contrastive training from [LAVIS](https://github.com/salesforce/LAVIS).

## BibTex

```bibtex
@inproceedings{xiao2024grounding,
    Author = {Zilin Xiao and Ming Gong and Paola Cascante-Bonilla and Xingyao Zhang and Jie Wu and Vicente Ordonez},
    Title = {Grounding Language Models for Visual Entity Recognition},
    Year = {2024},
    Eprint = {arXiv:2402.18695},
}
```
