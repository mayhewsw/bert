#$ -cwd
#$ -o /mnt/nlpgridio3/data/mayhew/bert_stuff/logs/out
#$ -e /mnt/nlpgridio3/data/mayhew/bert_stuff/logs/err
date

source activate tf

python create_pretraining_data.py \
       --input_file=${INPUT} \
       --output_file=$BERT_BASE_DIR/data/$(basename ${INPUT}).tfrecord \
       --vocab_file=$BERT_BASE_DIR/vocab.txt \
       --do_lower_case=False \
       --max_seq_length=128 \
       --do_whole_word_mask=True \
       --max_predictions_per_seq=20 \
       --masked_lm_prob=0.15 \
       --random_seed=12345 \
       --dupe_factor=5

date
