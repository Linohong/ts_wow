MODEL='roberta-large'
DATADIR='dataset/wizard_kat'
OUT_PREFIX='ts/checkpoints/ts_POC_do_tot'
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1

OUTPUT=${OUT_PREFIX}
PREFIX=train
python run_ts.py \
    --model_name_or_path $MODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --task 'Wizard_LowResource' \
    --train_prefix $PREFIX \
    --eval_prefix 'test_seen' \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_kno_length 64 \
    --max_num_kno 80 \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --true_topic_for_kno_sent \
    --save_steps 9000 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --output_dir $OUTPUT \
    --overwrite_output_dir \
    --do_tot \
    --true_topic_for_kno_sent \
    --fp16 
