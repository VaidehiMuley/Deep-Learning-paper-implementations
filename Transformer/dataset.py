import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([self.tokenizer_src.token_to_id('SOS')], dtype = torch.int64)
        self.eos_token = torch.Tensor([self.tokenizer_src.token_to_id('EOS')], dtype = torch.int64)
        self.pad_token = torch.Tensor([self.tokenizer_src.token_to_id('PAD')], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds(idx)
        src_sentence = src_target_pair['translation'][self.src_lang]
        tgt_sentence = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_sentence).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_sentence).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length is less than the number of tokens in the sentence")
        
        ## create the inputs
        
        # encoder input = [SOS] + enc input text + [EOS] + [PAD]

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype = torch.int64),
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype = torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
        ])

        assert encoder_input.size[0] == self.seq_len, "Encoder input length not equal to max seq len after padding"
        assert decoder_input.size[0] == self.seq_len, "Decoder input length not equal to max seq len after padding"
        assert label.size[0] == self.seq_len, "label length not equal to max seq len after padding"

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #[1,1,seq_len] mask the padding tokens - i/p to attention
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()  & self.causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_sentence,
            "tgt_text": tgt_sentence
        }
    
    def causal_mask(self,seq_len):
        mask = torch.triu(torch.ones(1,seq_len,seq_len), diagonal = 1).type(torch.int)
        return mask == 0





        