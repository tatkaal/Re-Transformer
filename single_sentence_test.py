import torch
import sentencepiece as spm
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from retransformer import ReTransformer  # Replace 'your_model_file' with the actual file name where your model is defined

### Hyperparameters ###
d_model = 512 # Number of features in the hidden state
num_heads = 8 # Number of heads in the multi-head attention layers
num_encoder_layers = 6  # As per the paper
num_decoder_layers = 2  # As per the paper
batch_size = 128
dropout = 0.1  # Dropout probability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Initialize SentencePiece models (assuming you have them loaded)
sp_src = spm.SentencePieceProcessor(model_file='src_only.model')
sp_tgt = spm.SentencePieceProcessor(model_file='tgt_only.model')

# Load vocabularies (assuming you have them saved)
# Replace these lines with your actual vocab loading code
SRC_vocab = torch.load('SRC_vocab.pth')
TGT_vocab = torch.load('TGT_vocab.pth')

print('TGT_vocab type is : ',type(TGT_vocab))

# Load the trained model
model_path = 'best_model_fold_3_6.2959.pth'  # Replace with your actual model path
model = ReTransformer(SRC_vocab, TGT_vocab, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout)
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to create padding mask
def create_padding_mask(seq):
    return (seq != 0).unsqueeze(1).unsqueeze(2)

# Function to prepare data and make predictions
def make_prediction(input_sentence):
    # Tokenize and numericalize the input sentence
    src_data = sp_src.encode(input_sentence, out_type=str)
    src_data = ['<s>'] + src_data + ['</s>']
    src_data = torch.tensor([SRC_vocab[token] for token in src_data], dtype=torch.long).unsqueeze(0)
    
    # Create source mask
    src_mask = create_padding_mask(src_data)
    
    # Initialize target tensor with '<s>' token
    tgt_data = torch.tensor([TGT_vocab['<s>']], dtype=torch.long).unsqueeze(0)
    
    for i in range(50):  # Assuming a maximum length of 50 for the target sequence
        tgt_mask = create_padding_mask(tgt_data)
        
        # Forward pass
        with torch.no_grad():
            output = model(src_data, tgt_data, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Get the token with highest probability as prediction for the next token
        next_token = output.argmax(2)[:, -1].item()
        
        # Append prediction to target tensor
        tgt_data = torch.cat([tgt_data, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
        
        # If the '</s>' token is generated, break the loop
        if next_token == TGT_vocab['</s>']:
            break
    
    # Decode the target tensor into a string
    decoded_tgt = sp_tgt.decode_ids(tgt_data[0].tolist())
    
    return ' '.join(decoded_tgt)

# Example usage
input_sentence = "God will save us all."
predicted_output = make_prediction(input_sentence)
print(f"Predicted Output: {predicted_output}")
