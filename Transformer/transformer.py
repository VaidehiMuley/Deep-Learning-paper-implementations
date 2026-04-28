import torch.nn as nn
from encoder import (Encoder, 
                     InputEmbedding, 
                     PositionEncoding,
                     MultiHeadAttention,
                     FeedForward,
                     EncoderBlock)
from decoder import (Decoder, 
                     ProjectLayer,
                     DecoderBlock)


class Transformer(nn.Mudule):
    def __init__(self, 
                encoder : Encoder, 
                decoder : Decoder, 
                src_embed : InputEmbedding,
                tgt_embed : InputEmbedding,
                src_pos : PositionEncoding,
                tgt_pos : PositionEncoding,
                proj_layer : ProjectLayer):
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src
    
    def decoder(self, encoder_output, src_mask,tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt =  self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt
    
    def project(self, x):
        return self.proj_layer(x)
    
def build_transformer(src_vocab_size : int,
                        tgt_vocab_size : int,
                        src_seq_len : int,
                        tgt_seq_len : int,
                        d_model: int = 512,
                        N: int = 6,
                        h: int = 8,
                        dropout: float = 0.1,
                        d_ff: int = 2048
                        ):
    
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    src_pos = PositionEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionEncoding(d_model, tgt_seq_len, dropout)

    # Encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff , dropout)
        encoder_block = EncoderBlock(encoder_self_attention,feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder Block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    ## Define the encoder and decoder
    encoder = nn.ModuleList(encoder_blocks)
    decoder = nn.ModuleList(decoder_blocks)

    ## Define the projection layer
    projection_layer = ProjectLayer(d_model, tgt_vocab_size)

    # create transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the transformer params
    for p in transformer.params():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer








