import torch
import torch.nn as nn
from encoder import (MultiHeadAttention, 
                     FeedForward,
                     ResidualNetwork)

class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block: MultiHeadAttention,
                    cross_attention_block : MultiHeadAttention,
                    feed_forward_block: FeedForward,
                    droput):
        
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualNetwork(droput) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        ## self attention
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        ## cross attention
        x = self.residual_connection[1](x, lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forward_block(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
    
class ProjectLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj_layer = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        ## Batch, seq_len, d_model --> Batch, seq_len, vocab_size
        return torch.log_softmax(self.proj_layer(x), dim = -1)



    