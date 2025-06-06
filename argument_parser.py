import argparse

parser = argparse.ArgumentParser(description = "Train a seq-seq Encoder Decoder based architecture for the task of Transliteration")

parser.add_argument('-b', '--batch_size', 
                  type = int, default = 128, 
                  help = 'Batch size')


parser.add_argument('-n', '--native', 
                  type = str, default = 'hi', 
                  help = 'Choices : bn, gu, hi, kn, ml, mr, pa, sd, si, ta, te, ur')

parser.add_argument('-m', '--model', 
                  type = str, default = "lstm", 
                  help = 'Choices : rnn, lstm, gru')

parser.add_argument('-eed', '--enc_embedding_dim', 
                  type = int, default = 64, 
                  help = 'Encoder Embedding dim')

parser.add_argument('-ed', '--enc_dim', 
                  type = int, default = 512, 
                  help = 'Encoder dim')

parser.add_argument('-ne', '--n_encoders', 
                  type = int, default = 1, 
                  help = 'Number of Encoders')

parser.add_argument('-ded', '--dec_embedding_dim', 
                  type = int, default = 64, 
                  help = 'Decoder Embedding dim')

parser.add_argument('-dd', '--dec_dim', 
                  type = int, default = 512, 
                  help = 'Decoder dim')

parser.add_argument('-nd', '--n_decoders', 
                  type = int, default = 1, 
                  help = 'Number of Decoders')

parser.add_argument('-ld', '--linear_dim', 
                  type = int, default = 2048, 
                  help = 'Linear Dim')

parser.add_argument('-do', '--dropout_rate', 
                  type = float, default = 0.4, 
                  help = 'Dropout rate')

parser.add_argument('-e', '--epochs', 
                  type = int, default = 20, 
                  help = 'Number of Epochs')

parser.add_argument('-ua', '--use_attn', 
                  type = bool, default = False, 
                  help = 'Set True to use attention')


parser.add_argument('-lc', '--log_connectivity', 
                  type = bool, default = False, 
                  help = 'Set True to log the connectivity diagram')


parser.add_argument('-mom', '--momentum', 
                  type = float, default = 0.9359397750924344,
                  help = 'Momentum to be used by the optimizer')

parser.add_argument('-k', '--beam_size', 
                  type = int, default = 3, 
                  help = 'K for Beam search')

parser.add_argument('-a', '--activation', 
                  type = str, default = "relu", 
                  help = 'Choices for activation : ReLU, Tanh, GeLU')

parser.add_argument('-o', '--optimizer', 
                  type = str, default = 'adamw',
                  help = 'Choices for optimizers : adam, adamw, rmsprop, adamax')

parser.add_argument('-lr', '--learning_rate', 
                  type = float, default = 0.003190691348613236,
                  help = 'Learning rate for optimizer')
                  
parser.add_argument('--use_v2', action='store_true', help='Use attention config version 2')

parser.add_argument('--use_test', action='store_true', help='Final evaluation on test data')

parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')

parser.add_argument('-we', '--wandb_entity', 
                  type = str, default = 'da24d008-iit-madras' ,
                  help = 'Wandb Entity used to track experiments in the Weights & Biases dashboard')

parser.add_argument('-wp', '--wandb_project', 
                  type = str, default = 'da6401-assignment3' ,
                  help = 'Project name used to track experiments in Weights & Biases dashboard')

parser.add_argument('--wandb_sweep', action='store_true', help='Enable W&B sweep')

parser.add_argument('--sweep_id', type = str, help = "Sweep ID", default = None)
