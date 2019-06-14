import sentencepiece as spm
import random
import tokenization

fname = sys.argv[1]

MODEL_PREFIX = "sentpiece"
VOC_SIZE = 1500
SUBSAMPLE_SIZE = 128000
NUM_PLACEHOLDERS = 256

tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
outfname = fname + ".tmp"
with open(fname) as f, open(outfname, "w") as out:
    for line in f:
        outline = " ".join(tokenizer.tokenize(line)) 
        out.write(outline + "\n")
        
SPM_COMMAND = ('--input={} --model_prefix={} --normalization_rule_name=identity '
               '--vocab_size={} '
               '--shuffle_input_sentence=true --model_type=unigram '
               '--bos_id=-1 --eos_id=-1').format(
    outfname, MODEL_PREFIX,
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

VOC_FNAME = "vocab.txt"

with open(VOC_FNAME, "w") as fo:
    for token in bert_vocab:
        fo.write(token+"\n")

