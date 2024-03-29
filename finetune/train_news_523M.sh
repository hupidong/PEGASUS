python train.py \
    --task=news \
    --eval_checkpoint=0 \
    --model_name_or_path=IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese \
    --train_file=../../../../../kg/gen/news_summary/v2/train/train.jsonl \
    --eval_file=../../../../../kg/gen/news_summary/v2/train/dev.jsonl \
    --save_dir=checkpoints/news_summary/Randeng-Pegasus-523M-Summary-Chinese \
    --init_checkpoint=checkpoints/news_summary/Randeng-Pegasus-523M-Summary-Chinese/model_best \
    --max_source_length=1024 \
    --max_target_length=200 \
    --epoch=20 \
    --logging_steps=10 \
    --save_interval=1 \
    --eval_interval=20 \
    --train_batch_size=2 \
    --eval_batch_size=15 \
    --learning_rate=5e-5 \
    --warmup_proportion=0.02 \
    --weight_decay=0.01 \
    --device=gpu:0 \
    --do_lower_case=0 \
    --metric_weights 1.0 0.9 1.0 1.5