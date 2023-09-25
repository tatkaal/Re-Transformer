import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator
from sacrebleu import corpus_bleu
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import sentencepiece as spm
import argparse
from retransformer import ReTransformer

### Hyperparameters ###
d_model = 512 # Number of features in the hidden state
num_heads = 8 # Number of heads in the multi-head attention layers
num_encoder_layers = 6  # As per the paper
num_decoder_layers = 2  # As per the paper
batch_size = 128
dropout = 0.1  # Dropout probability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Add the test model path here
best_model_path='best_model_fold_3_6.2959.pth'

# Set all 3 variables to None if you want to use the entire dataset
test_sample_size = 5

# Initialize SentencePiece models
sp_src = spm.SentencePieceProcessor(model_file='src_only.model')
sp_tgt = spm.SentencePieceProcessor(model_file='tgt_only.model')

def build_unified_vocab(filenames):
    all_src_data = []
    all_tgt_data = []
    for filename in filenames:
        df = pd.read_csv(filename)
        src_data = [sp_src.encode(item, out_type=str) for item in df['src']]
        tgt_data = [sp_tgt.encode(item, out_type=str) for item in df['tgt']]
        all_src_data.extend(src_data)
        all_tgt_data.extend(tgt_data)
    
    SRC_vocab = build_vocab_from_iterator(all_src_data, specials=['<unk>', '<s>', '</s>'])
    SRC_vocab.set_default_index(SRC_vocab['<unk>'])
    TGT_vocab = build_vocab_from_iterator(all_tgt_data, specials=['<unk>', '<s>', '</s>'])
    TGT_vocab.set_default_index(TGT_vocab['<unk>'])
    
    return SRC_vocab, TGT_vocab

# Build unified vocab
SRC_vocab, TGT_vocab = build_unified_vocab(['training.csv', 'validation.csv', 'testing.csv'])
print('Vocab built successfully')

# Save the vocabularies
torch.save(SRC_vocab, 'SRC_vocab.pth')
torch.save(TGT_vocab, 'TGT_vocab.pth')

# print the first 10 words in TGT_vocab
print('First 100 words in target vocab TGT_vocab : ', [TGT_vocab.get_itos()[i] for i in range(100)])

def create_padding_mask(seq):
    return (seq != 0).unsqueeze(1).unsqueeze(2)  # shape [batch_size, 1, 1, seq_len]

# New function to create DataLoader from CSV
def csv_to_dataloader_and_vocab(filename, batch_size, SRC_vocab, TGT_vocab, sample_size=None):
    df = pd.read_csv(filename)
    if sample_size:
        df = df.sample(n=sample_size)  # Randomly sample 'sample_size' rows from the DataFrame
    src_data = [sp_src.encode(item, out_type=str) for item in df['src']]
    tgt_data = [sp_tgt.encode(item, out_type=str) for item in df['tgt']]
    
    # Add special tokens
    src_data = [['<s>'] + item + ['</s>'] for item in src_data]
    tgt_data = [['<s>'] + item + ['</s>'] for item in tgt_data]

    # Convert to numerical and pad sequences
    src_data = [torch.tensor([SRC_vocab[token] for token in item], dtype=torch.long) for item in src_data]
    tgt_data = [torch.tensor([TGT_vocab[token] for token in item], dtype=torch.long) for item in tgt_data]
    
    src_data = pad_sequence(src_data, batch_first=True, padding_value=SRC_vocab['<pad>'])
    tgt_data = pad_sequence(tgt_data, batch_first=True, padding_value=TGT_vocab['<pad>'])
    
    src_mask = create_padding_mask(src_data)
    tgt_mask = create_padding_mask(tgt_data)

    dataloader = DataLoader(TensorDataset(src_data, tgt_data, src_mask, tgt_mask), batch_size=batch_size)
    
    return dataloader

# DataLoaders and Vocab
test_dataloader = csv_to_dataloader_and_vocab('testing.csv', batch_size, SRC_vocab, TGT_vocab, test_sample_size)
print('DataLoaders and Vocab created successfully')


model = ReTransformer(SRC_vocab, TGT_vocab, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout)
model.load_state_dict(torch.load(best_model_path))
model.to(device)  # Move model to the device

model.eval()
all_trg = []
all_translated_trg = []

for i, (src, tgt, src_mask, tgt_mask) in enumerate(test_dataloader):
    src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)
    
    # Create mask for src and tgt (assuming padding token is 0)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2).to(device)

    output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)  # Include masks here

    output = output.argmax(dim=2)

    for i in range(src.size(0)):  # Loop over each sentence in the batch
        sentence_trg = [TGT_vocab.get_itos()[idx] for idx in tgt[i].cpu().numpy() if idx != 0]  # Exclude padding
        sentence_translated_trg = [TGT_vocab.get_itos()[idx] for idx in output[i].cpu().numpy() if idx != 0]  # Exclude padding

        # Convert to normal sentence
        subword_sentence_trg = "".join(sentence_trg).replace("▁", " ").strip()
        subword_sentence_translated_trg = "".join(sentence_translated_trg).replace("▁", " ").strip()

        # Decode subwords back to original sentence
        original_sentence_trg = sp_tgt.decode(subword_sentence_trg)
        
        all_trg.append([original_sentence_trg])
        all_translated_trg.append(subword_sentence_translated_trg)

# Calculate BLEU score
bleu_score = corpus_bleu(all_translated_trg, all_trg).score

print(f'BLEU score = {bleu_score}')
print("Sample translated targets:", all_translated_trg[:5])

df_test = pd.read_csv('testing.csv')
actual_targets = df_test['tgt'][:5].tolist()
print('Actual Targets : ', actual_targets)

# Create a DataFrame and save to CSV
df = pd.DataFrame({
    'Target': all_trg,
    'Predicted': all_translated_trg
})
df.to_csv('test_predictions.csv', index=False)