import pandas as pd
import sentencepiece as spm

# # Read the CSV file
df = pd.read_csv('training.csv')

# Extract the 'tgt' column and save it to a text file
with open('target_sentences.txt', 'w', encoding='utf-8') as f:
    for sentence in df['tgt']:
        f.write(sentence + '\n')

#increase vocab_size when using actual dataset
spm.SentencePieceTrainer.train(input='target_sentences.txt', model_prefix='tgt_only', vocab_size=16000, pad_id=3, unk_id=2, bos_id=0, eos_id=1)