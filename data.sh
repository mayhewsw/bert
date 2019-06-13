
set -e

SHARD_DIR=/tmp/bert_shards/

PRC_DATA_FPATH=/nlp/data/mayhew/bert_stuff/uig.txt

mkdir -p $SHARD_DIR
split -a 4 -l 25600 -d $PRC_DATA_FPATH $SHARD_DIR/shard_

BERT_BASE_DIR=/nlp/data/mayhew/bert_stuff/uig

mkdir -p $BERT_BASE_DIR/data

for f in $SHARD_DIR/*;
do
    echo $f;
    python create_pretraining_data.py \
           --input_file=${f} \
           --output_file=$BERT_BASE_DIR/data/$(basename ${f}).tfrecord \
           --vocab_file=$BERT_BASE_DIR/vocab.txt \
           --do_lower_case=False \
           --max_seq_length=128 \
           --max_predictions_per_seq=20 \
           --masked_lm_prob=0.15 \
           --random_seed=12345 \
           --dupe_factor=5
done

#rm -rf $SHARD_DIR
