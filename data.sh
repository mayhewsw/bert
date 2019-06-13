set -e

export BERT_BASE_DIR=/nlp/data/mayhew/bert_stuff/uig
SHARD_DIR=$BERT_BASE_DIR/shards

PRC_DATA_FPATH=/nlp/data/mayhew/bert_stuff/uig.txt

mkdir -p $SHARD_DIR
split -a 4 -l 25000 -d $PRC_DATA_FPATH $SHARD_DIR/shard_

mkdir -p $BERT_BASE_DIR/data

for f in $SHARD_DIR/*;
do
    export FNAME=$f
    qsub -v INPUT=$f,BERT_BASE_DIR=$BERT_BASE_DIR  -N $(basename $f) queue_data.sh
done

#rm -rf $SHARD_DIR
