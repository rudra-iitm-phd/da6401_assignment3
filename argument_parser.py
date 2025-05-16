import argparse

parser = argparse.ArgumentParser(description = "Train a seq-seq Encoder Decoder based architecture for the task of Transliteration")

parser.add_argument('-b', '--batch_size', 
                  type = int, default = 256, 
                  help = 'Batch size')


parser.add_argument('-n', '--native', 
                  type = str, default = 'hi', 
                  help = 'Choices : bn, gu, hi, kn, ml, mr, pa, sd, si, ta, te, ur')

parser.add_argument('-m', '--model', 
                  type = str, default = "lstm", 
                  help = 'Choices : rnn, lstm, gru')

parser.add_argument('-eed', '--enc_embedding_dim', 
                  type = int, default = 128, 
                  help = 'Encoder Embedding dim')

parser.add_argument('-ed', '--enc_dim', 
                  type = int, default = 128, 
                  help = 'Encoder dim')

parser.add_argument('-ne', '--n_encoders', 
                  type = int, default = 3, 
                  help = 'Number of Encoders')

parser.add_argument('-ded', '--dec_embedding_dim', 
                  type = int, default = 128, 
                  help = 'Decoder Embedding dim')

parser.add_argument('-dd', '--dec_dim', 
                  type = int, default = 128, 
                  help = 'Decoder dim')

parser.add_argument('-nd', '--n_decoders', 
                  type = int, default = 3, 
                  help = 'Number of Decoders')

parser.add_argument('-ld', '--linear_dim', 
                  type = int, default = 512, 
                  help = 'Linear Dim')

parser.add_argument('-do', '--dropout_rate', 
                  type = float, default = 0, 
                  help = 'Dropout rate')