


python run_pretraining.py \
       --input_file=/tmp/*.tfrecord \
       --output_dir=/tmp/pretraining_output \
       --do_train=True \
       --do_eval=True \
       --bert_config_file=$BERT_BASE_DIR/bert_config.json \
       --train_batch_size=32 \
       --max_seq_length=128 \
       --use_tpue = True \
       --max_predictions_per_seq=20 \
       --num_train_steps=10000 \
       --num_warmup_steps=10 \
       --learning_rate=2e-5


#--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
