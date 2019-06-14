#$ -cwd
#$ -o voc_out
#$ -e voc_err
#$ -l h_vmem=128G
date

source activate anlp8.2

python mkvocab.py /mnt/nlpgridio3/data/mayhew/bert_stuff/uzugen-shuf.txt.tmp /mnt/nlpgridio3/data/mayhew/bert_stuff/uzugen/vocab.txt

date
