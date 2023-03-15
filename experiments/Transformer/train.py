import argparse
import time
import os

import torch
from torch import nn
from torch.optim import Adam

from utils.optimizers_and_distributions import CustomLRAdamOptimizer, LabelSmoothingDistribution
from transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
import utils.utils as utils
from utils.constants import *
import random
import numpy as np
import mesa as ms

random_seed = 526
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# Global vars for logging purposes
num_of_trg_tokens_processed = 0
train_losses = []
val_losses = []
bleu_scores = []
li=[]
total_li=[]
global_train_step, global_val_step = [0, 0]
train_mem = 0

# Simple decorator function so that I don't have to pass these arguments every time I call get_train_val_loop
def get_train_val_loop(transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time_start):

    def train_val_loop(is_train, token_ids_loader, epoch):
        global num_of_trg_tokens_processed, global_train_step, global_val_step, train_losses, val_losses, train_mem, li
        if is_train:
            transformer.train()
            global total_li
        else:
            transformer.eval()

        device = next(transformer.parameters()).device
        #
        # Main loop - start of the CORE PART
        #
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)
            # log because the KL loss expects log probabilities (just an implementation detail)
            pre_allocated = torch.cuda.memory_allocated(device) / 1024 /1024  
            predicted_log_distributions, li = transformer(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
            post_allocated = torch.cuda.memory_allocated(device) / 1024 /1024  
            act_mem = (post_allocated - pre_allocated)
            if act_mem > train_mem:
                train_mem = act_mem
            smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)  # these are regular probabilities
            
            if is_train:
                custom_lr_optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph
                
            loss = kl_div_loss(predicted_log_distributions, smooth_target_distributions)

            if is_train:
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                custom_lr_optimizer.step()  # apply the gradients to weights
            # End of CORE PART

            #
            # Logging and metrics
            #
            if is_train:
                global_train_step += 1
                num_of_trg_tokens_processed += num_trg_tokens

                if training_config['enable_tensorboard']:
                    train_losses.append(loss.item())
                    
                if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                    print(f'time elapsed= {(time.time() - time_start):.2f} [s] '
                          f'| epoch={epoch + 1} | batch= {batch_idx + 1} '
                          f'| target tokens/batch= {num_of_trg_tokens_processed / training_config["console_log_freq"]}'
                          f'| loss= {loss.item()}'
                          f'| training memory= {train_mem} [Mib]')
                    num_of_trg_tokens_processed = 0
                
                if training_config['get_li']: 
                    if batch_idx%100==0 and training_config['learning_rule'] !='bp':
                        total_li += random.sample(li, 1000)
                '''
                # Save model checkpoint
                if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0 and batch_idx == 0:
                    ckpt_model_name = f"transformer_ckpt_epoch_{epoch + 1}.pth"
                    torch.save(utils.get_training_state(training_config, transformer), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
                '''
            else:
                global_val_step += 1

                if training_config['enable_tensorboard']:
                    val_losses.append(loss.item())
            
    return train_val_loop


def train_transformer(training_config):
    device = torch.device(training_config['device'])  # checking whether you have a GPU, I hope so!

    # Step 1: Prepare data loaders
    train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        training_config['dataset_path'],
        training_config['language_direction'],
        training_config['dataset_name'],
        training_config['batch_size'],
        device)

    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)

    # Step 2: Prepare the model (original transformer) and push to GPU
    if training_config['model'] == 'baseline':
        transformer = Transformer(
            model_dimension=BASELINE_MODEL_DIMENSION,
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
            number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
            dropout_probability=BASELINE_MODEL_DROPOUT_PROB,
            learning_rule = training_config['learning_rule'],
            gcp = training_config['gcp'],
            get_li = training_config['get_li'],
            mesa = training_config['mesa']
        )
    
    else:
        transformer = Transformer(
            model_dimension=BIG_MODEL_DIMENSION,
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            number_of_heads=BIG_MODEL_NUMBER_OF_HEADS,
            number_of_layers=BIG_MODEL_NUMBER_OF_LAYERS,
            dropout_probability=BIG_MODEL_DROPOUT_PROB,
            learning_rule = training_config['learning_rule'],
            gcp = training_config['gcp'],
            get_li=training_config['get_li'],
            mesa = training_config['mesa']
        )
        
    if args.mesa:
        ms.policy.convert_by_num_groups(transformer, BASELINE_MODEL_NUMBER_OF_HEADS)
        ms.policy.deploy_on_init(transformer, args.mesa_policy, verbose=print, override_verbose=False)
    
    transformer.to(device)
    
    
    print('net memory : ' + str(torch.cuda.memory_allocated(device) / 1024 /1024))
    print(transformer)

    # Step 3: Prepare other training related utilities
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')  # gives better BLEU score than "mean"

    # Makes smooth target distributions as opposed to conventional one-hot distributions
    # My feeling is that this is a really dummy and arbitrary heuristic but time will tell.
    label_smoothing = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device)
    if training_config['model'] == 'big':
        label_smoothing = LabelSmoothingDistribution(BIG_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device)
    
    # Check out playground.py for an intuitive visualization of how the LR changes with time/training steps, easy stuff.
    custom_lr_optimizer = CustomLRAdamOptimizer(
                Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
                BASELINE_MODEL_DIMENSION,
                training_config['num_warmup_steps']
            )
    if training_config['model'] == 'big':
        custom_lr_optimizer = CustomLRAdamOptimizer(
                    Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
                    BIG_MODEL_DIMENSION,
                    training_config['num_warmup_steps']
                )
        
    
    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    train_val_loop = get_train_val_loop(transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time.time())

    # Step 4: Start the training
    max_bleu = 0
    save_file_name = 'Baseline_Transformer'
    if training_config['model'] == 'big':
        save_file_name = 'Big_Transformer'
    
    if training_config['learning_rule'] == 'asa1':
        save_file_name += '_asa1'    
    if training_config['learning_rule'] == 'asa2':
        save_file_name += '_asa2'
    if training_config['learning_rule'] == 'asa3':
        save_file_name += '_asa3'
    if training_config['learning_rule'] == 'asa4':
        save_file_name += '_asa4'
        
    if training_config['mesa']:
        save_file_name += '_mesa'
    
    for epoch in range(training_config['num_of_epochs']):
        # Training loop
        train_val_loop(is_train=True, token_ids_loader=train_token_ids_loader, epoch=epoch)
        # Validation loop
        with torch.no_grad():
            train_val_loop(is_train=False, token_ids_loader=val_token_ids_loader, epoch=epoch)

            bleu_score = utils.calculate_bleu_score(transformer, val_token_ids_loader, trg_field_processor)
            if training_config['enable_tensorboard']:
                bleu_scores.append(bleu_score)
            # Save the transformer which has best bleu
            if bleu_score > max_bleu:
                max_bleu = bleu_score
                #torch.save(utils.get_training_state(training_config, bleu_score, transformer), os.path.join(BINARIES_PATH, utils.get_available_binary_name()))
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(utils.get_training_state(training_config, bleu_score, transformer), "checkpoint/" + save_file_name + '.pth')
                print('save models')
        if not os.path.isdir('results'):
            os.mkdir('results')
        np.save("results/" + save_file_name +'-train_loss.npy', train_losses)
        np.save("results/" + save_file_name +'-val_loss.npy', val_losses)
        np.save("results/" + save_file_name +'-bleu.npy', bleu_scores)
        if training_config['get_li']:
            np.save("results/" + save_file_name +'-li.npy', total_li)
                
if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    num_warmup_steps = 4000

    #
    # Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    # According to the paper I infered that the baseline was trained for ~19 epochs on the WMT-14 dataset and I got
    # nice returns up to epoch ~20 on IWSLT as well (nice round number)
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=20)
    # You should adjust this for your particular machine (I have RTX 2080 with 8 GBs of VRAM so 1500 fits nicely!)
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Data related args
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)
    parser.add_argument("--dataset_path", type=str, help='dataset path for Transformer', default=DATA_DIR_PATH)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=True)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    parser.add_argument('--learning_rule', type=str, choices = ['bp', 'asa1', 'asa2', 'asa3', 'asa4'], default='bp',  help='learning rule to use (default: bp)')
    parser.add_argument("--gcp", type=bool, help="gcp for self-attention", default=False)
    parser.add_argument("--get_li", type=bool, help="get learning indicator", default=False)
    parser.add_argument('--model', type=str, choices = ['baseline', 'big'], default='baseline',  help='model to use (default: baseline)')    
    parser.add_argument('--device', type= int, default = 0, help='device_num')
    parser.add_argument("--mesa", type=bool, help="apply mesa", default=False)
    parser.add_argument("--mesa_policy", type=str, help='path of mesa policy' )
    
    
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['num_warmup_steps'] = num_warmup_steps

    # Train the original transformer model
    train_transformer(training_config)
