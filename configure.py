from model import *
from data import *
from torch.utils.data import DataLoader
from model import DynamicSeq2Seq
import torch.nn as nn
class Configure:
      def __init__(self, script):
            self.script = script
            self.native = self.script['native']
            self.batch_size = self.script['batch_size']

            self.train_dataset = NativeLatinDataset(lang = self.native, data = 'train')
            self.train_dl = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)

            self.val_dataset = NativeLatinDataset(lang = self.native, data = 'dev')
            self.val_dl = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = True)

            self.test_dataset = NativeLatinDataset(lang = self.native, data = 'test')
            self.test_dl = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = True)

            self.latin_char2idx = self.train_dataset.latin_char2idx
            self.native_char2idx = self.train_dataset.native_char2idx

            self.activation = self.script['activation'].lower()

            self.model_type = self.script["model"].lower()

            self.encoder_embedding_input_dim = len(self.latin_char2idx)
            self.encoder_embedding_output_dim = self.script['enc_embedding_dim']
            self.enc_ouput_dim = self.script['enc_dim']
            self.n_encoders = self.script['n_encoders']

            self.decoder_embedding_input_dim = len(self.native_char2idx)
            self.decoder_embedding_output_dim = self.script['dec_embedding_dim']
            self.dec_ouput_dim = self.script['dec_dim']
            self.n_decoders = self.script['n_decoders']

            self.linear_dim = self.script['linear_dim']
            self.dropout_rate = self.script['dropout_rate']
            
            self.optimizer_mappings = {
                  'adam':torch.optim.Adam,
                  'adamax':torch.optim.Adamax,
                  'rmsprop':torch.optim.RMSprop,
                  'adamw':torch.optim.AdamW
            }

            self.optim = self.script['optimizer'].lower()
            self.learning_rate = self.script['learning_rate']
            # self.weight_decay = self.script['weight_decay']
            self.momentum = self.script['momentum']

            self.model = None

      def get_datasets(self):
            return self.train_dataset, self.val_dataset, self.test_dataset
      
      def get_dataloders(self):
            return self.train_dl, self.val_dl, self.test_dl

      def get_model(self):
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
                  self.dropout_rate
                  )

            self.model = model

            return model

      def get_optimzer(self):
            if self.optim != 'rmsprop':
                  return self.optimizer_mappings[self.optim](self.model.parameters(),
                                                            lr = self.learning_rate, 
                                                            betas = (self.momentum, 0.999))
            else:
                  return self.optimizer_mappings[self.optim](self.model.parameters(),
                                                            lr = self.learning_rate, 
                                                            momentum = self.momentum)


