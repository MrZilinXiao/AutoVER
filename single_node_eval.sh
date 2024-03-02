MODEL_BASE="lmsys/vicuna-7b-v1.5"  # only works for lora-model
MODEL_PATH="./checkpoints/llava-v1.5-7b-0.2-finalv1-rice"
PRETRAIN_MM_MLP_ADAPTER="None"
LOG_PATH="./oven_eval_logs_7b_rice_0.2_ret/"
CONSTRAINED_DECODE_TYPE="raw_ret"
./create_screens.sh eval0 "cd /home/zx51/scratch_code/v_entity; CUDA_VISIBLE_DEVICES=0 python eval_oven_entity_ret.py --val_jsonl_path ./ms_jsonl_dataset/oven_entity_val_0.jsonl ./ms_jsonl_dataset/oven_entity_test_labeled_rice_0.jsonl ./ms_jsonl_dataset/oven_query_val_rice_0.jsonl ./ms_jsonl_dataset/oven_query_test_labeled_rice_0.jsonl ./ms_jsonl_dataset/oven_human_labeled_rice_0.jsonl --fake_dist 0 --model_base ${MODEL_BASE} --model_path ${MODEL_PATH} --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} --log_path ${LOG_PATH} --constrained_decode_type ${CONSTRAINED_DECODE_TYPE} --img_shards_path /home/zx51/dataset/entity_oven"
./create_screens.sh eval1 "cd /home/zx51/scratch_code/v_entity; CUDA_VISIBLE_DEVICES=1 python eval_oven_entity_ret.py --val_jsonl_path ./ms_jsonl_dataset/oven_entity_val_1.jsonl ./ms_jsonl_dataset/oven_entity_test_labeled_rice_1.jsonl ./ms_jsonl_dataset/oven_query_val_rice_1.jsonl ./ms_jsonl_dataset/oven_query_test_labeled_rice_1.jsonl ./ms_jsonl_dataset/oven_human_labeled_rice_1.jsonl --fake_dist 1 --model_base ${MODEL_BASE} --model_path ${MODEL_PATH} --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} --log_path ${LOG_PATH} --constrained_decode_type ${CONSTRAINED_DECODE_TYPE} --img_shards_path /home/zx51/dataset/entity_oven"
./create_screens.sh eval2 "cd /home/zx51/scratch_code/v_entity; CUDA_VISIBLE_DEVICES=2 python eval_oven_entity_ret.py --val_jsonl_path ./ms_jsonl_dataset/oven_entity_val_2.jsonl ./ms_jsonl_dataset/oven_entity_test_labeled_rice_2.jsonl ./ms_jsonl_dataset/oven_query_val_rice_2.jsonl ./ms_jsonl_dataset/oven_query_test_labeled_rice_2.jsonl ./ms_jsonl_dataset/oven_human_labeled_rice_2.jsonl --fake_dist 2 --model_base ${MODEL_BASE} --model_path ${MODEL_PATH} --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} --log_path ${LOG_PATH} --constrained_decode_type ${CONSTRAINED_DECODE_TYPE} --img_shards_path /home/zx51/dataset/entity_oven"
./create_screens.sh eval3 "cd /home/zx51/scratch_code/v_entity; CUDA_VISIBLE_DEVICES=3 python eval_oven_entity_ret.py --val_jsonl_path ./ms_jsonl_dataset/oven_entity_val_3.jsonl ./ms_jsonl_dataset/oven_entity_test_labeled_rice_3.jsonl ./ms_jsonl_dataset/oven_query_val_rice_3.jsonl ./ms_jsonl_dataset/oven_query_test_labeled_rice_3.jsonl ./ms_jsonl_dataset/oven_human_labeled_rice_3.jsonl --fake_dist 3 --model_base ${MODEL_BASE} --model_path ${MODEL_PATH} --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} --log_path ${LOG_PATH} --constrained_decode_type ${CONSTRAINED_DECODE_TYPE} --img_shards_path /home/zx51/dataset/entity_oven"
./create_screens.sh eval4 "cd /home/zx51/scratch_code/v_entity; CUDA_VISIBLE_DEVICES=4 python eval_oven_entity_ret.py --val_jsonl_path ./ms_jsonl_dataset/oven_entity_val_4.jsonl ./ms_jsonl_dataset/oven_entity_test_labeled_rice_4.jsonl ./ms_jsonl_dataset/oven_query_val_rice_4.jsonl ./ms_jsonl_dataset/oven_query_test_labeled_rice_4.jsonl ./ms_jsonl_dataset/oven_human_labeled_rice_4.jsonl --fake_dist 4 --model_base ${MODEL_BASE} --model_path ${MODEL_PATH} --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} --log_path ${LOG_PATH} --constrained_decode_type ${CONSTRAINED_DECODE_TYPE} --img_shards_path /home/zx51/dataset/entity_oven"
./create_screens.sh eval5 "cd /home/zx51/scratch_code/v_entity; CUDA_VISIBLE_DEVICES=5 python eval_oven_entity_ret.py --val_jsonl_path ./ms_jsonl_dataset/oven_entity_val_5.jsonl ./ms_jsonl_dataset/oven_entity_test_labeled_rice_5.jsonl ./ms_jsonl_dataset/oven_query_val_rice_5.jsonl ./ms_jsonl_dataset/oven_query_test_labeled_rice_5.jsonl ./ms_jsonl_dataset/oven_human_labeled_rice_5.jsonl --fake_dist 5 --model_base ${MODEL_BASE} --model_path ${MODEL_PATH} --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} --log_path ${LOG_PATH} --constrained_decode_type ${CONSTRAINED_DECODE_TYPE} --img_shards_path /home/zx51/dataset/entity_oven"
./create_screens.sh eval6 "cd /home/zx51/scratch_code/v_entity; CUDA_VISIBLE_DEVICES=6 python eval_oven_entity_ret.py --val_jsonl_path ./ms_jsonl_dataset/oven_entity_val_6.jsonl ./ms_jsonl_dataset/oven_entity_test_labeled_rice_6.jsonl ./ms_jsonl_dataset/oven_query_val_rice_6.jsonl ./ms_jsonl_dataset/oven_query_test_labeled_rice_6.jsonl ./ms_jsonl_dataset/oven_human_labeled_rice_6.jsonl --fake_dist 6 --model_base ${MODEL_BASE} --model_path ${MODEL_PATH} --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} --log_path ${LOG_PATH} --constrained_decode_type ${CONSTRAINED_DECODE_TYPE} --img_shards_path /home/zx51/dataset/entity_oven"
./create_screens.sh eval7 "cd /home/zx51/scratch_code/v_entity; CUDA_VISIBLE_DEVICES=7 python eval_oven_entity_ret.py --val_jsonl_path ./ms_jsonl_dataset/oven_entity_val_7.jsonl ./ms_jsonl_dataset/oven_entity_test_labeled_rice_7.jsonl ./ms_jsonl_dataset/oven_query_val_rice_7.jsonl ./ms_jsonl_dataset/oven_query_test_labeled_rice_7.jsonl ./ms_jsonl_dataset/oven_human_labeled_rice_7.jsonl --fake_dist 7 --model_base ${MODEL_BASE} --model_path ${MODEL_PATH} --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} --log_path ${LOG_PATH} --constrained_decode_type ${CONSTRAINED_DECODE_TYPE} --img_shards_path /home/zx51/dataset/entity_oven"