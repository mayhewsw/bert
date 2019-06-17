import sentencepiece as spm
import random
import tokenization
import os
import sys

filedir = sys.argv[1]
outvocab = sys.argv[2]

MODEL_PREFIX = "sentpiece"
VOC_SIZE = 30000
NUM_PLACEHOLDERS = 256

fnames = filter(lambda f: f.endswith(".tok"), os.listdir(filedir))
fnames = list(map(lambda f: os.path.join(filedir, f), fnames))
comma_sep_fnames = ",".join(fnames)

SPM_COMMAND = ('--input={} --model_prefix={} --normalization_rule_name=identity '
               '--vocab_size={} --num_threads=24 --input_sentence_size=2000000 '
               '--shuffle_input_sentence=true --model_type=unigram '
               '--bos_id=-1 --eos_id=-1').format(
    comma_sep_fnames, MODEL_PREFIX,
    VOC_SIZE - NUM_PLACEHOLDERS)

spm.SentencePieceTrainer.Train(SPM_COMMAND)

def read_sentencepiece_vocab(filepath):
    voc = []
    with open(filepath, encoding='utf-8') as fi:
        for line in fi:
            voc.append(line.split("\t")[0])
    # skip the first <unk> token
    voc = voc[1:]
    return voc
        
snt_vocab = read_sentencepiece_vocab("{}.vocab".format(MODEL_PREFIX))
print("Learnt vocab size: {}".format(len(snt_vocab)))
print("Sample tokens: {}".format(random.sample(snt_vocab, 10)))


def parse_sentencepiece_token(token):
    if token.startswith("▁"):
        return token[1:]
    else:
        return "##" + token

bert_vocab = list(map(parse_sentencepiece_token, snt_vocab))

ctrl_symbols = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
bert_vocab = ctrl_symbols + bert_vocab

bert_vocab += ["[UNUSED_{}]".format(i) for i in range(VOC_SIZE - len(bert_vocab))]
print(len(bert_vocab))

with open(outvocab, "w") as fo:
    for token in bert_vocab:
        fo.write(token+"\n")

print("Write to", outvocab)
        

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

print("Writing files to", os.path.dirname(outvocab))
with open("{}/bert_config.json".format(os.path.dirname(outvocab)), "w") as fo:
  json.dump(bert_base_config, fo, indent=2)


