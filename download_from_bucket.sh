
GC_SDK=/mnt/castor/seas_home/m/mayhew/IdeaProjects/google-cloud-sdk/

export MODEL_DIR=/nlp/data/mayhew/bert_stuff

#$GC_SDK/bin/gsutil cp gs://uig_bert/pretraining_output/model.ckpt-127000* $MODEL_DIR/pretrained_models

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
                        $MODEL_DIR/pretrained_models/model.ckpt-127000 \
                        $MODEL_DIR/uzugen/bert_config.json \
                        $MODEL_DIR/pretrained_models/pytorch_model.bin
