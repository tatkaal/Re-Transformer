import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator
from sacrebleu import corpus_bleu
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import sentencepiece as spm
import argparse
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from retransformer import ReTransformer
from torch.optim.lr_scheduler import OneCycleLR

### Hyperparameters ###
d_model = 512 # Number of features in the hidden state
num_heads = 8 # Number of heads in the multi-head attention layers
num_encoder_layers = 6  # As per the paper
num_decoder_layers = 2  # As per the paper
dropout = 0.1
batch_size = 64
num_epochs = 300
# weight_decay = 0
learning_rate = 0.01
# n_folds = 3  # Number of folds for cross-validation
patience = 50  # Number of num_epochs to wait before early stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set all 3 variables to None if you want to use the entire dataset
train_sample_size = 5000
valid_sample_size = 500
test_sample_size = 100

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="A script with a Boolean argument")

# Add a Boolean argument
parser.add_argument("--hyperparametertuning", action="store_true", help="A Boolean flag")

# Parse the command-line arguments
args = parser.parse_args()

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
print(f"SRC Vocabulary size: {len(SRC_vocab)}")
print(f"TGT Vocabulary size: {len(TGT_vocab)}")

# Save the vocabularies
torch.save(SRC_vocab, 'SRC_vocab.pth')
torch.save(TGT_vocab, 'TGT_vocab.pth')

# Check if special tokens exist in the vocabularies
assert '<s>' in SRC_vocab and '</s>' in SRC_vocab, "Special tokens <s> or </s> not found in SRC_vocab"
assert '<s>' in TGT_vocab and '</s>' in TGT_vocab, "Special tokens <s> or </s> not found in TGT_vocab"

# print the first 10 words in TGT_vocab
print('First 100 words in target vocab TGT_vocab : ', [TGT_vocab.get_itos()[i] for i in range(100)])

def create_padding_mask(seq):
    return (seq != 0).unsqueeze(1).unsqueeze(2)  # shape [batch_size, 1, 1, seq_len]

# Sample check for create_padding_mask function
sample_seq = torch.tensor([[1, 2, 0, 4], [1, 0, 0, 4]])
sample_mask = create_padding_mask(sample_seq)
assert sample_mask.shape[-1] == sample_seq.shape[-1], "Last dimension mismatch between sample sequence and mask"

def inspect_activation(name):
    def hook(model, input, output):
        print(f"{name}: Mean={output.data.mean():.2f}, Std={output.data.std():.2f}")
    return hook

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
train_dataloader = csv_to_dataloader_and_vocab('training.csv', batch_size, SRC_vocab, TGT_vocab, train_sample_size)
valid_dataloader = csv_to_dataloader_and_vocab('validation.csv', batch_size, SRC_vocab, TGT_vocab, valid_sample_size)
test_dataloader = csv_to_dataloader_and_vocab('testing.csv', batch_size, SRC_vocab, TGT_vocab, test_sample_size)
print('DataLoaders and Vocab created successfully')

if args.hyperparametertuning:
    print('performing hyperparameter tuning.....')
    from hyperparametertuner import main as hyperparameter_tuner
    hyperparameter_tuner(SRC_vocab, TGT_vocab, train_dataloader, valid_dataloader, device, num_epochs)
    exit()

kf = KFold(n_splits=n_folds, shuffle=True)

print("Training started...")
best_global_val_loss = float('inf')
best_model_path = ''
log_file = 'training_logs.txt'

# for fold, (train_index, valid_index) in enumerate(kf.split(train_dataloader.dataset)):
#     print(f'FOLD {fold + 1}')

# Initialize the best validation loss to infinity for each fold
best_val_loss = float('inf')

model = ReTransformer(SRC_vocab, TGT_vocab, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout)

# Move the model to the GPU
model = model.to(device)

# Loss and Optimizer
pad_idx = TGT_vocab['<pad>']
assert pad_idx == 0, f"Padding index is {pad_idx}, but CrossEntropyLoss is set to ignore index 0"
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning Rate Scheduler
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1)
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_dataloader))

# Initialize patience counter
patience_counter = 0

# # Register hooks (add these lines after defining your model and before training)
# model.encoder[0].register_forward_hook(inspect_activation('Encoder Layer 1'))
# model.decoder[0].register_forward_hook(inspect_activation('Decoder Layer 1'))

# Training Loop with Validation
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for i, (src, tgt, src_mask, tgt_mask) in enumerate(train_dataloader):
        # if i == 0:  # Only print for the first batch
        #     print("Source:", src)
        #     print("Target:", tgt)
        #     print("Source Mask:", src_mask)
        #     print("Target Mask:", tgt_mask)
        src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # Compute loss
        output_dim = output.shape[-1]
        output = output[:-1].view(-1, output_dim)
        tgt = tgt[1:].view(-1)

        loss = criterion(output, tgt)

        # Backward pass and optimization
        loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.data.norm(2).item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # Step the scheduler
        scheduler.step()

        total_train_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}")

    # Validation part
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for i, (src, tgt, src_mask, tgt_mask) in enumerate(valid_dataloader):
            src, tgt, src_mask, tgt_mask = src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device)

            # Forward pass
            output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

            # Compute loss
            output_dim = output.shape[-1]
            output = output[:-1].view(-1, output_dim)
            tgt = tgt[1:].view(-1)
            val_loss = criterion(output, tgt)
            total_val_loss += val_loss.item()

            print(f"Validation, Batch [{i + 1}/{len(valid_dataloader)}], Validation Loss: {val_loss.item():.4f}")

    current_lr = optimizer.param_groups[0]['lr']
    avg_val_loss = total_val_loss / len(valid_dataloader)
    # Log the epoch results to the file
    with open(log_file, 'a') as f:
        # f.write(f"FOLD {fold+1}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_train_loss/len(train_dataloader):.4f}, Validation Loss: {avg_val_loss:.4f}, Current LR: {current_lr}\n")
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_train_loss/len(train_dataloader):.4f}, Validation Loss: {avg_val_loss:.4f}, Current LR: {current_lr}\n")
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss / len(train_dataloader):.4f}, Validation Loss: {avg_val_loss:.4f}, Current LR: {current_lr}")

    # scheduler.step(avg_val_loss)

    # Check if this is the best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        with open(log_file, 'a') as f:
            f.write(f"Best validation loss improved, saving model...\n")
        print("Best validation loss improved, saving model...")
        
        # Save the model for each fold
        # model_path = f'best_model_fold_{fold + 1}_{best_val_loss:.4f}.pth'
        model_path = f'best_model_{best_val_loss:.4f}.pth'
        torch.save(model.state_dict(), model_path)
        if best_val_loss < best_global_val_loss:
            best_global_val_loss = best_val_loss
            best_model_path = model_path

    else:
        patience_counter += 1
        if patience_counter >= patience:
            with open(log_file, 'a') as f:
                # f.write(f"Early stopping triggered at fold {fold + 1}, epoch {epoch + 1}. Best Validation Loss: {best_val_loss}\n")
                f.write(f"Early stopping triggered at epoch {epoch + 1}. Best Validation Loss: {best_val_loss}\n")
            # print(f"Early stopping triggered at fold {fold + 1}, epoch {epoch + 1}. Best Validation Loss: {best_val_loss}")
            print(f"Early stopping triggered at epoch {epoch + 1}. Best Validation Loss: {best_val_loss}")
            break

with open(log_file, 'a') as f:
    f.write(f"Training completed successfully!\n")
print('Training completed successfully!')

# Model Evaluation part on test dataset

# Load the best model back into memory
model.load_state_dict(torch.load(best_model_path))

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