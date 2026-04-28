import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,vocab_size, d_model):
        """Input embedding layer for the Transformer model.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model - we will use 512
        """

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self,x):
        """
        Forward pass for the Input Embedding
        """
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):
    """Encoding input into contextual embeddings"""
    
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create the positional encodings
        self.position_encodings = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                self.position_encodings[pos, i] = math.sin(pos/ (10000 ** (i/ self.d_model))) # Even
                self.position_encodings[pos, i+1] = math.cos(pos/ (10000 ** (i/ self.d_model))) # Odd
        self.position_encodings = self.position_encodings.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('positional_encodings', self.position_encodings)

    def forward(self, x):
        """
        Forward pass for the Position Encoding
        """
        x = x + self.positional_encodings[:, :x.shape[1], :].requires_grad_(False) #All batches, upto this particular seq_len, all embedding dimensions
        return self.dropout(x)

class LayerNormalisation(nn.Module):
    ## Normalise the d_model for each token independently.
    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim =True)
        return self.alpha * (x - mean)/ (std + self. eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dff: int , dropout) -> None:
        self.d_model = d_model
        self.dff = dff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.d_model, self.dff)
        self.linear2 = nn.Linear(self.dff, self.d_model)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.Relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h ==0, "d_model should be divisible by head"
        self.d_k = self.d_model // h
        self.w_q =  nn.Linear(self.d_model, self.d_model)
        self.w_k =  nn.Linear(self.d_model, self.d_model)
        self.w_v =  nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        ## attention score = softmax(Q. K.T // sqrt(d_k))
        ## key = Batch,h,seq_len,d_k --> batch,h,d_k,seq_len
        ## attention_scores = batch,h,d_k,d_k

        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) // math.sqrt(d_k)

        # attention scores =batch,h,seq_len,d_k X batch,h,d_k,seq_len =  batch,h,seq_len,seq_len
        if mask:
            attention_scores.masked_fill(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) ## batch,h,seq_len,seq
        if dropout:
            attention_scores = dropout(attention_scores)
        
        # batch,h,seq_len,seq_len X batch,h,seq_len,d_k
        return attention_scores @ value


    def forward(self, q,k,v, mask):

        query = self.w_q(q) # Batch_num, seq_len, d_model
        key = self.w_k(k)
        value = self.w_v(v)

        # Batch, seq_len, d_model --> Batch, seq_len, h, d_k --> batch, h, seq_len, d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # concatenate all the heads together
        ## batch,h,seq_len,d_k --> batch,seq_len,d_model
        x.transpose(1,2).contiguous().view(x.shape[0], x.shape[1] ,self.d_k * self.h)

        # batch,seq_len,d_model --> batch,seq_len,d_model
        return self.w_o(x)
    
class ResidualNetwork(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation()

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))
    
    
class EncoderBlock(nn.Module):
    def __init__(self, 
                attention_block : MultiHeadAttention,
                feedforward_block : FeedForward,
                dropout: float) -> None:
        
        super().__init__()
        self.attention_block = attention_block
        self.feedforward_block = feedforward_block
        self.residual_block = nn.ModuleList([ResidualNetwork(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_block[0](x, lambda x: self.attention_block(x,x,x,src_mask))
        x = self.residual_block[1](x, self.feedforward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x , mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

    
    




        











