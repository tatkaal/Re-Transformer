import pandas as pd
import sentencepiece as spm

# # Read the CSV file
df = pd.read_csv('training.csv')

# Extract the 'tgt' column and save it to a text file
with open('target_sentences.txt', 'w', encoding='utf-8') as f:
    for sentence in df['tgt'][:5000]:
        f.write(sentence + '\n')

#increase vocab_size when using actual dataset
spm.SentencePieceTrainer.train('--input=target_sentences.txt --model_prefix=tgt_only --vocab_size=3000 --model_type=unigram --pad_id=1 --unk_id=3 --bos_id=0 --eos_id=2 --max_sentence_length=256 --num_threads=8 --character_coverage=1')