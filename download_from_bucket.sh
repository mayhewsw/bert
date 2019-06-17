
GC_SDK=/mnt/castor/seas_home/m/mayhew/IdeaProjects/google-cloud-sdk/

export MODEL_DIR=/nlp/data/mayhew/bert_stuff/uzugen/

STEP=2373000
$GC_SDK/bin/gsutil cp gs://uig_bert_wp/pretraining_output/model.ckpt-${STEP}* $MODEL_DIR/pretrained_models

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
                        $MODEL_DIR/pretrained_models/model.ckpt-${STEP} \
                        $MODEL_DIR/bert_config.json \
                        $MODEL_DIR/pretrained_models/pytorch_model.bin

tar czvf $MODEL_DIR/pretrained_models/bert-model-${STEP}.tar.gz $MODEL_DIR/bert_config.json $MODEL_DIR/pretrained_models/pytorch_model.bin
