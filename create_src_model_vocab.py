import pandas as pd
import sentencepiece as spm

# # Read the CSV file
df = pd.read_csv('training.csv')

# Extract the 'src' column and save it to a text file
with open('source_sentences.txt', 'w', encoding='utf-8') as f:
    for sentence in df['src']:
        f.write(sentence + '\n')

spm.SentencePieceTrainer.train(input='source_sentences.txt', model_prefix='src_only', vocab_size=16000, pad_id=3, unk_id=2, bos_id=0, eos_id=1)