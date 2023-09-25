import math
import torch
import torch.nn as nn
import torch.optim as optim
from queue import PriorityQueue

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Expand dimensions to: [1, max_len, d_model]
        pe = self.pe.unsqueeze(0)
        # Cut off at seq_len and add positional encoding
        x = x + pe[:, :x.size(1), :]
        return x

# Define the Multi-Head Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "Embedding size needs to be divisible by num_heads"

        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)
        key = key.reshape(N, key_len, self.num_heads, self.head_dim)
        value = value.reshape(N, value_len, self.num_heads, self.head_dim)

        # Scaled Dot-Product Attention
        scores = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        attention = torch.nn.functional.softmax(scores, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
            N, query_len, self.d_model
        )

        out = self.fc_out(out)
        return out

# Re-Transformer Encoder Layer with Delayed Non-Linear Transformation
class ReTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, delay_nl=2):
        super(ReTransformerEncoderLayer, self).__init__()
        self.self_attn1 = MultiHeadAttention(d_model, num_heads)
        self.self_attn2 = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.delay_nl = delay_nl

    def forward(self, x, layer_num):
        attn_output1 = self.self_attn1(x, x, x)
        attn_output2 = self.self_attn2(attn_output1, attn_output1, attn_output1)
        
        if layer_num >= self.delay_nl:
            x = x + self.dropout(attn_output1)
            x = self.norm1(x)
            x = x + self.dropout(attn_output2)
            x = self.norm2(x)
            
            # Apply feed-forward layer only for specific layers
            if layer_num % 2 == 0:
                ff_output = self.feed_forward(x)
                x = x + self.dropout(ff_output)
                x = self.norm3(x)
        else:
            x = x + self.dropout(attn_output2)
            x = self.norm2(x)
        
        return x


# Re-Transformer Decoder Layer
class ReTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(ReTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)  # New attention layer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        attn_output = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        
        attn_output = self.enc_dec_attn(x, enc_output, enc_output)  # New attention layer
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        
        return x

# Complete Re-Transformer Model
class ReTransformer(nn.Module):
    def __init__(self, SRC_vocab, TGT_vocab, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(ReTransformer, self).__init__()
        self.SRC_vocab = SRC_vocab  # if needed
        self.TGT_vocab = TGT_vocab  # Store it as a class attribute
        self.src_embedding = nn.Embedding(len(SRC_vocab), d_model)
        self.tgt_embedding = nn.Embedding(len(TGT_vocab), d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList(
            [ReTransformerEncoderLayer(d_model, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [ReTransformerDecoderLayer(d_model, num_heads, dropout) for _ in range(num_decoder_layers)]
        )
        self.fc_out = nn.Linear(d_model, len(TGT_vocab))

    def forward(self, src, tgt=None):
        src = self.src_embedding(src)  # Embed source sequences
        src = self.pos_encoder(src)

        for layer_num, layer in enumerate(self.encoder):
            src = layer(src, layer_num)

        if tgt is not None:
            tgt = self.tgt_embedding(tgt)  # Embed target sequences only if not None
            tgt = self.pos_encoder(tgt)
            # Use the encoder's output in the decoder
            for layer in self.decoder:
                tgt = layer(tgt, src)  # Add the encoder's output to the decoder's input
            output = self.fc_out(tgt)
            return output
        else:
            pass  # for beam search