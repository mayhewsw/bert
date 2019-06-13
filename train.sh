
B=gs://uig_bert

python run_pretraining.py \
       --input_file=$B/data/*.tfrecord \
       --output_dir=$B/pretraining_output \
       --do_train=True \
       --do_eval=True \
       --bert_config_file=$B/bert_config.json \
       --train_batch_size=32 \
       --max_seq_length=128 \
       --use_tpu = True \
       --max_predictions_per_seq=20 \
       --num_train_steps=10000 \
       --num_warmup_steps=10 \
       --learning_rate=2e-5


#--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
