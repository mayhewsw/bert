# Run this script on the nlpgrid

export BERT_BASE_DIR=/nlp/data/mayhew/bert_stuff/uzugen
SHARD_DIR=$BERT_BASE_DIR/shards

# IMPORTANT: if training on multiple languages, this input file needs to be document shuffled.
# Use shuffle_doc.sh
PRC_DATA_FPATH=/nlp/data/mayhew/bert_stuff/uzugen-shuf.txt

if [ ! -d $SHARD_DIR ];
then
    mkdir $SHARD_DIR
    split -a 4 -l 25000 -d $PRC_DATA_FPATH $SHARD_DIR/shard_
fi

# Delete all prior tmp files.
rm -rf $SHARD_DIR/*.tok

# tokenize all
for f in $SHARD_DIR/*;
do
    ./qsub-script -N $(basename $f) -o /dev/null -e /dev/null "source activate anlp8.2; python tok.py ${f}"
done
