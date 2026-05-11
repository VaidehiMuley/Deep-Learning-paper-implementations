import torch
import torch.nn as nn
from torch.utils.data  import random_split, DataLoader, Dataset
from pathlib import Path

from datasets import load_dataset
from config import get_config, get_weights_file_path, latest_weights_file_path
from tokenizer import Tokenizer
from tokenizer.models import WordLevel
from tokenizer.trainers import WordLevelTrainer
from tokenizer.pre_tokenizers import Whitespace
from dataset import BilingualDataset
from transformer import build_transformer



def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unl_token = '[UNK]'))
        tokenizer.pretokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens= [['UNK'], 'PAD', 'EOS','SOS'], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer = trainer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config['lang_src']}-{config['lang_tgt']}', split = 'train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    

    ## Split into train and val sets
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = int(0.1 * len(ds_raw))
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    # For seq_len , find the max len of the input and output text
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config]['lang_tgt']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of the input text: {max_len_src}')
    print(f'Max length of the output text: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size= config['batch_size'], shuffle= True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle = True) # Process 1 at a time

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model =  build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f"Using {device} device")

    Path(config['model_folder']).mkdir(parents = True, exist_ok= True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())

    ## Define optimizer and loss
    optimizer = torch.optim.adam(model.parameters(), lr = config['lr'], eps = 1e-9)
    ## Label smoothing --> so that model isn't very sure thus less over fitting
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    


    
    









        
