MODEL='facebook/bart-large'
DATADIR='dataset/wizard_kat'
OUT_PREFIX='bartcat_lino2'
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1

OUTPUT=checkpoints/${OUT_PREFIX}
PREFIX=train

for seed in 42
do
    for rat in 0 
    do
    echo "seed: ${seed} "
    echo "rat: ${rat} "
        if [ $rat -eq 0 ]; then
            OUTPUT=checkpoints/${OUT_PREFIX}
            PREFIX=train
        else
            OUTPUT=checkpoints/${OUT_PREFIX}_${rat}
            PREFIX=train_$rat
        fi
        python run_BART_lino.py \
            --model_name_or_path $MODEL \
            --data_dir $DATADIR \
            --cache_dir 'cached' \
            --task 'Wizard_LowResource' \
            --train_prefix $PREFIX \
            --eval_prefix 'test_seen' \
            --max_source_length 256 \
            --max_target_length 64 \
            --max_kno_length 64 \
            --max_num_kno 40 \
            --do_eval \
            --num_train_epochs 5 \
            --save_steps 9000 \
            --per_gpu_train_batch_size 8 \
            --per_gpu_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --output_dir $OUTPUT \
            --overwrite_output_dir \
            --seed ${seed} \
            --ts_dir 'checkpoints/ts/checkpoints/ts_POC/checkpoint-54000' \
            --fp16
    done
done
