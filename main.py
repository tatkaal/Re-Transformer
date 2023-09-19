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

### Hyperparameters ###
d_model = 512 # Number of features in the hidden state
num_heads = 8 # Number of heads in the multi-head attention layers
num_encoder_layers = 6  # As per the paper
num_decoder_layers = 2  # As per the paper
dropout = 0.1
batch_size = 32
num_epochs = 100
weight_decay = 0.0001
learning_rate = 0.0001
n_folds = 5  # Number of folds for cross-validation
patience = 5  # Number of num_epochs to wait before early stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

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
    
    return SRC_vocab, TGT_vocab

# Build unified vocab
SRC_vocab, TGT_vocab = build_unified_vocab(['training.csv', 'validation.csv', 'testing.csv'])
print('Vocab built successfully')
# print(f"SRC Vocabulary size: {len(SRC_vocab)}")
# print(f"TGT Vocabulary size: {len(TGT_vocab)}")

# print the first 10 words in TGT_vocab
print('First 100 words in target vocab TGT_vocab : ', [TGT_vocab.get_itos()[i] for i in range(100)])

# New function to create DataLoader from CSV
def csv_to_dataloader_and_vocab(filename, batch_size, SRC_vocab, TGT_vocab):
    df = pd.read_csv(filename)
    src_data = [sp_src.encode(item, out_type=str) for item in df['src']]
    tgt_data = [sp_tgt.encode(item, out_type=str) for item in df['tgt']]
    
    # Convert to numerical and pad sequences
    src_data = [torch.tensor([SRC_vocab[token] for token in item], dtype=torch.long) for item in src_data]
    tgt_data = [torch.tensor([TGT_vocab[token] for token in item], dtype=torch.long) for item in tgt_data]
    
    src_data = pad_sequence(src_data, batch_first=True, padding_value=0)
    tgt_data = pad_sequence(tgt_data, batch_first=True, padding_value=0)
    
    dataloader = DataLoader(TensorDataset(src_data, tgt_data), batch_size=batch_size, shuffle=True)
    
    return dataloader

# DataLoaders and Vocab
train_dataloader = csv_to_dataloader_and_vocab('training.csv', batch_size, SRC_vocab, TGT_vocab)
valid_dataloader = csv_to_dataloader_and_vocab('validation.csv', batch_size, SRC_vocab, TGT_vocab)
test_dataloader = csv_to_dataloader_and_vocab('testing.csv', batch_size, SRC_vocab, TGT_vocab)
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

for fold, (train_index, valid_index) in enumerate(kf.split(train_dataloader.dataset)):
    print(f'FOLD {fold + 1}')

    # Initialize the best validation loss to infinity for each fold
    best_val_loss = float('inf')
    
    model = ReTransformer(SRC_vocab, TGT_vocab, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout)
    
    # Move the model to the GPU
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1)

    # Initialize patience counter
    patience_counter = 0

    # Training Loop with Validation
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for i, (src, tgt) in enumerate(train_dataloader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(src, tgt)

            # Compute loss
            output_dim = output.shape[-1]
            output = output[:-1].view(-1, output_dim)
            tgt = tgt[1:].view(-1)

            loss = criterion(output, tgt)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}")

        # Validation part
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for i, (src, tgt) in enumerate(valid_dataloader):
                src, tgt = src.to(device), tgt.to(device)

                # Forward pass
                output = model(src, tgt)

                # Compute loss
                output_dim = output.shape[-1]
                output = output[:-1].view(-1, output_dim)
                tgt = tgt[1:].view(-1)
                val_loss = criterion(output, tgt)
                total_val_loss += val_loss.item()

                print(f"Validation, Batch [{i + 1}/{len(valid_dataloader)}], Validation Loss: {val_loss.item():.4f}")

        avg_val_loss = total_val_loss / len(valid_dataloader)
        # Log the epoch results to the file
        with open(log_file, 'a') as f:
            f.write(f"FOLD {fold+1}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_train_loss/len(train_dataloader):.4f}, Validation Loss: {avg_val_loss:.4f}, Current LR: {current_lr}\n")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss / len(train_dataloader):.4f}, Validation Loss: {avg_val_loss:.4f}, Current LR: {current_lr}")

        scheduler.step(avg_val_loss)

        # Check if this is the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            with open(log_file, 'a') as f:
                f.write(f"Best validation loss improved, saving model...\n")
            print("Best validation loss improved, saving model...")
            
            # Save the model for each fold
            model_path = f'best_model_fold_{fold + 1}_{best_val_loss:.4f}.pth'
            torch.save(model.state_dict(), model_path)
            if best_val_loss < best_global_val_loss:
                best_global_val_loss = best_val_loss
                best_model_path = model_path

        else:
            patience_counter += 1
            if patience_counter >= patience:
                with open(log_file, 'a') as f:
                    f.write(f"Early stopping triggered at fold {fold + 1}, epoch {epoch + 1}. Best Validation Loss: {best_val_loss}\n")
                print(f"Early stopping triggered at fold {fold + 1}, epoch {epoch + 1}. Best Validation Loss: {best_val_loss}")
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

for i, batch in enumerate(test_dataloader):
    src, trg = batch # Unpack the batch into src and tgt
    src, trg = src.to(device), trg.to(device)
    output = model(src, trg)

    ##uncomment to do beam search instead but not working
    # output = model(src, None)

    output = output.argmax(dim=2)

    for i in range(src.size(0)):  # Loop over each sentence in the batch
        sentence_trg = [TGT_vocab.get_itos()[idx] for idx in trg[i].cpu().numpy() if idx != 0]  # Exclude padding
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
# print("Sample actual targets:", all_trg[:5])

df_test = pd.read_csv('testing.csv')
actual_targets = df_test['tgt'][:5].tolist()
print('Actual Targets : ', actual_targets)