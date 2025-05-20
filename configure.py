from model import *  # Import everything from model.py
from data import *  # Import everything from data.py
from torch.utils.data import DataLoader  # DataLoader to handle batching
from model import DynamicSeq2Seq  # Importing specific model class
import torch.nn as nn  # Neural network module

class Configure:
      def __init__(self, script):
            self.script = script  # Dictionary of configuration parameters
            self.native = self.script['native']  # Native language identifier
            self.batch_size = self.script['batch_size']  # Batch size for data loading

            # Initialize training dataset and dataloader
            self.train_dataset = NativeLatinDataset(lang = self.native, data = 'train')
            self.train_dl = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)

            # Initialize validation dataset and dataloader
            self.val_dataset = NativeLatinDataset(lang = self.native, data = 'dev')
            self.val_dl = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = True)

            # Initialize test dataset and dataloader
            self.test_dataset = NativeLatinDataset(lang = self.native, data = 'test')
            self.test_dl = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = True)

            # Character-to-index mappings for Latin and Native scripts
            self.latin_char2idx = self.train_dataset.latin_char2idx
            self.native_char2idx = self.train_dataset.native_char2idx

            self.activation = self.script['activation'].lower()  # Activation function name
            self.model_type = self.script["model"].lower()  # Model type: lstm, gru, or rnn
            self.use_attn = self.script['use_attn']  # Whether to use attention

            # Encoder configuration
            self.encoder_embedding_input_dim = len(self.latin_char2idx)
            self.encoder_embedding_output_dim = self.script['enc_embedding_dim']
            self.enc_ouput_dim = self.script['enc_dim']
            self.n_encoders = self.script['n_encoders']

            # Decoder configuration
            self.decoder_embedding_input_dim = len(self.native_char2idx)
            self.decoder_embedding_output_dim = self.script['dec_embedding_dim']
            self.dec_ouput_dim = self.script['dec_dim']
            self.n_decoders = self.script['n_decoders']

            self.linear_dim = self.script['linear_dim']  # Linear layer dimension
            self.dropout_rate = self.script['dropout_rate']  # Dropout rate
            
            # Mapping from optimizer names to PyTorch optimizer classes
            self.optimizer_mappings = {
                  'adam':torch.optim.Adam,
                  'adamax':torch.optim.Adamax,
                  'rmsprop':torch.optim.RMSprop,
                  'adamw':torch.optim.AdamW
            }

            self.optim = self.script['optimizer'].lower()  # Selected optimizer
            self.learning_rate = self.script['learning_rate']  # Learning rate
            # self.weight_decay = self.script['weight_decay']  # Optional weight decay (commented)
            self.momentum = self.script['momentum']  # Momentum for optimizer

            self.model = None  # Placeholder for model instance

      def get_datasets(self):
            return self.train_dataset, self.val_dataset, self.test_dataset  # Return all datasets
      
      def get_dataloders(self):
            return self.train_dl, self.val_dl, self.test_dl  # Return all dataloaders

      def get_model(self):
            # Initialize and return the model instance with all parameters
            model = DynamicSeq2Seq(
                  self.model_type,
                  self.activation,
                  self.encoder_embedding_input_dim, 
                  self.encoder_embedding_output_dim, 
                  self.enc_ouput_dim, 
                  self.n_encoders, 
                  self.decoder_embedding_input_dim,
                  self.decoder_embedding_output_dim,
                  self.dec_ouput_dim,
                  self.n_decoders,
                  self.linear_dim,
                  self.dropout_rate,
                  self.use_attn
                  )

            self.model = model  # Save model to instance

            return model  # Return model

      def get_optimzer(self):
            # Return configured optimizer instance using optimizer mapping
            if self.optim != 'rmsprop':
                  return self.optimizer_mappings[self.optim](self.model.parameters(),
                                                            lr = self.learning_rate, 
                                                            betas = (self.momentum, 0.999))
            else:
                  return self.optimizer_mappings[self.optim](self.model.parameters(),
                                                            lr = self.learning_rate, 
                                                            momentum = self.momentum)
