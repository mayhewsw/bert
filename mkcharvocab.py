import random
import sys
import json
#import tqdm
import os

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
MODEL_PREFIX = "tokenizer"
VOC_SIZE = 165
SUBSAMPLE_SIZE = 12800000
NUM_PLACEHOLDERS = 5

assert os.path.exists(OUTPUT_PATH)

voc = set()
with open(INPUT_PATH) as f:
    for i,line in enumerate(f):
        if i%10000 == 0:
            print(i)
        for c in line.decode("utf8").strip():
            voc.add(c)
            voc.add("##" + c)

bert_vocab = list(voc)

ctrl_symbols = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
bert_vocab = ctrl_symbols + bert_vocab
bert_vocab += ["[UNUSED_{}]".format(i) for i in range(VOC_SIZE - len(bert_vocab))]
print("Vocab size", len(bert_vocab))

VOC_FNAME = "vocab.txt" #@param {type:"string"}

bert_base_config = {
    "attention_probs_dropout_prob": 0.1, 
    "directionality": "bidi", 
    "hidden_act": "gelu", 
    "hidden_dropout_prob": 0.1, 
    "hidden_size": 768, 
    "initializer_range": 0.02, 
    "intermediate_size": 3072, 
    "max_position_embeddings": 512, 
    "num_attention_heads": 12, 
    "num_hidden_layers": 12, 
    "pooler_fc_size": 768, 
    "pooler_num_attention_heads": 12, 
    "pooler_num_fc_layers": 3, 
    "pooler_size_per_head": 128, 
    "pooler_type": "first_token_transform", 
    "type_vocab_size": 2, 
    "vocab_size": VOC_SIZE
}

print("Writing files to", OUTPUT_PATH)
with open("{}/bert_config.json".format(OUTPUT_PATH), "w") as fo:
  json.dump(bert_base_config, fo, indent=2)
  
with open("{}/{}".format(OUTPUT_PATH, VOC_FNAME), "w") as fo:
  for token in bert_vocab:
    fo.write(token.encode("utf8")+"\n")
