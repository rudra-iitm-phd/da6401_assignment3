import torch 
import torch.nn as nn
import numpy as np
class RNN(nn.Module):
      def __init__(self, encoder_input_dim, decoder_input_dim):
            super(RNN, self).__init__()
            self.enc_embedding = nn.Embedding(encoder_input_dim, 64)
            self.encoder = nn.RNN(64, 128, batch_first = True)

            self.dec_embedding = nn.Embedding(decoder_input_dim, 64)
            self.decoder = nn.RNN(64, 128, batch_first = True)

            self.fc = nn.Linear(128, decoder_input_dim)
            
      def forward(self, x):
            x = self.enc_embedding(x)
            output, hidden_state = self.encoder(x)
            dec_hidden = hidden_state
            decoder_input = torch.tensor([[3]] * x.size(0), device = 'mps') 
            outputs = []
            for t in range(20):
                  embed = self.dec_embedding(decoder_input)
                  dec_op, dec_hidden = self.decoder(embed, dec_hidden)
                  output = self.fc(dec_op.squeeze(1))
                  outputs.append(output)
                  decoder_input = output.argmax(-1).unsqueeze(1)

            return torch.stack(outputs, 1)


class DynamicRNN(nn.Module):
      def __init__(self, encoder_embedding_input_dim, encoder_embedding_output_dim, enc_ouput_dim, n_encoders, decoder_embedding_input_dim, decoder_embedding_output_dim, dec_ouput_dim, n_decoders, linear_dim, dropout_rate=0):
            super(DynamicRNN, self).__init__()

            self.enc_embedding_vocab_size = encoder_embedding_input_dim
            self.enc_embedding_dim = encoder_embedding_output_dim
            self.enc_dim = enc_ouput_dim
            self.n_encoders = n_encoders

            self.dec_embedding_vocab_size = decoder_embedding_input_dim
            self.dec_embedding_dim = decoder_embedding_output_dim
            self.dec_dim = dec_ouput_dim
            self.n_decoders = n_decoders

            self.linear_dim = linear_dim

            self.encoder_embedding = nn.Embedding(self.enc_embedding_vocab_size, self.enc_embedding_dim)
            self.encoder = nn.RNN(self.enc_embedding_dim, self.enc_dim, num_layers = self.n_encoders, batch_first = True)

            self.decoder_embedding = nn.Embedding(self.dec_embedding_vocab_size, self.dec_embedding_dim)
            self.decoder = nn.RNN(self.dec_embedding_dim, self.dec_dim, num_layers = self.n_decoders, batch_first = True)

            self.fc1 = nn.Linear(self.dec_dim, self.linear_dim)
            self.fc2 = nn.Linear(self.linear_dim, self.dec_embedding_vocab_size)

            self.project_encoder_hidden_to_decoder_hidden = nn.Linear(self.n_encoders * self.enc_dim, self.n_decoders * self.dec_dim)

      def forward(self, x, y=None):
            x = self.encoder_embedding(x)
            enc_output, enc_hidden_state = self.encoder(x) 
            # dim(hidden_state) = [n_encoders, batch_size, enc_output_dim]
            # dim(output) = [batch_size, max_len_tokens, enc_output_dim]
            if self.n_decoders == self.n_encoders:
                  dec_hidden = enc_hidden_state # This dimension match applies if and only if n_encoders == n_decoders
            else:
                  """
                  Projecting the encoder hidden state to decoder hidden state dimensions 
                  to ensure size mismatch when n_decoders != n_encoders

                  """
                  enc_hidden_state = enc_hidden_state.transpose(0, 1).contiguous()
                  enc_hidden_state = enc_hidden_state.view(enc_hidden_state.size(0), -1)

                  enc_hidden_state = self.project_encoder_hidden_to_decoder_hidden(enc_hidden_state)

                  dec_hidden = enc_hidden_state.view(enc_hidden_state.size(0), self.n_decoders, self.dec_dim)
                  dec_hidden = dec_hidden.transpose(0, 1).contiguous()


            dec_input = torch.tensor([[2]] * x.size(0), device = 'mps') # start tokens
            outputs = []
            for i in range(10):
                  dec_embed = self.decoder_embedding(dec_input)
                  dec_output, dec_hidden = self.decoder(dec_embed, dec_hidden)
                  output = self.fc1(dec_output.squeeze(1))
                  output = self.fc2(output)
                  outputs.append(output)
                  if y is not None:
                        dec_input = y[:,i]
                  else:
                        dec_input = output.argmax(-1).unsqueeze(1)
                  dec_input = output.argmax(-1).unsqueeze(1)
            
            return torch.stack(outputs, 1)


class DynamicLSTM(nn.Module):
      def __init__(self, encoder_embedding_input_dim, encoder_embedding_output_dim, enc_ouput_dim, n_encoders, decoder_embedding_input_dim, decoder_embedding_output_dim, dec_ouput_dim, n_decoders, linear_dim, dropout_rate):
            super(DynamicLSTM, self).__init__()

            self.enc_embedding_vocab_size = encoder_embedding_input_dim
            self.enc_embedding_dim = encoder_embedding_output_dim
            self.enc_dim = enc_ouput_dim
            self.n_encoders = n_encoders

            self.dec_embedding_vocab_size = decoder_embedding_input_dim
            self.dec_embedding_dim = decoder_embedding_output_dim
            self.dec_dim = dec_ouput_dim
            self.n_decoders = n_decoders

            self.linear_dim = linear_dim

            self.encoder_embedding = nn.Embedding(self.enc_embedding_vocab_size, self.enc_embedding_dim)
            self.encoder = nn.LSTM(self.enc_embedding_dim, self.enc_dim, num_layers = self.n_encoders, batch_first = True)

            self.decoder_embedding = nn.Embedding(self.dec_embedding_vocab_size, self.dec_embedding_dim)
            self.decoder = nn.LSTM(self.dec_embedding_dim, self.dec_dim, num_layers = self.n_decoders, batch_first = True)

            self.fc1 = nn.Linear(self.dec_dim, self.linear_dim)
            self.fc2 = nn.Linear(self.linear_dim, self.dec_embedding_vocab_size)

            self.project_encoder_hidden_to_decoder_hidden = nn.Linear(self.n_encoders * self.enc_dim, self.n_decoders * self.dec_dim)
            self.project_encoder_cell_to_decoder_cell = nn.Linear(self.n_encoders * self.enc_dim, self.n_decoders * self.dec_dim)

      def forward(self, x, y=None):
            x = self.encoder_embedding(x)
            output, (enc_hidden_state, enc_cell_state) = self.encoder(x)
            # note : hidden_state_dim == cell_state_dim
            if self.n_decoders == self.n_encoders:
                  dec_hidden = enc_hidden_state
                  dec_cell = enc_cell_state
            else:
                  """
                  Projecting the encoder hidden state to decoder hidden state dimensions 
                  to ensure size mismatch when n_decoders != n_encoders
                  
                  """
                  enc_hidden_state = enc_hidden_state.transpose(0, 1).contiguous()
                  enc_hidden_state = enc_hidden_state.view(enc_hidden_state.size(0), -1)

                  enc_hidden_state = self.project_encoder_hidden_to_decoder_hidden(enc_hidden_state)

                  dec_hidden = enc_hidden_state.view(enc_hidden_state.size(0), self.n_decoders, self.dec_dim)
                  dec_hidden = dec_hidden.transpose(0, 1).contiguous()

                  """
                  Projecting the encoder cell state to decoder cell state dimensions 
                  to ensure size mismatch when n_decoders != n_encoders
                  
                  """
                  enc_cell_state = enc_cell_state.transpose(0, 1).contiguous()
                  enc_cell_state = enc_cell_state.view(enc_cell_state.size(0), -1)

                  enc_cell_state = self.project_encoder_cell_to_decoder_cell(enc_cell_state)

                  dec_cell = enc_cell_state.view(enc_cell_state.size(0), self.n_decoders, self.dec_dim)
                  dec_cell = dec_cell.transpose(0, 1).contiguous()

            
            dec_input = torch.tensor([[2]] * x.size(0), device = 'mps')
            
            outputs = []
            for i in range(10):
                  dec_embed = self.decoder_embedding(dec_input)
                  dec_output, (dec_hidden, dec_cell) = self.decoder(dec_embed, (dec_hidden, dec_cell))
                  output = self.fc1(dec_output.squeeze(1))
                  output = self.fc2(output)
                  outputs.append(output)
                  if y is not None:
                        dec_input = y[:,i].view(-1, 1)
                  else:
                        dec_input = output.argmax(-1).unsqueeze(1)
            return torch.stack(outputs, 1)

      
class DynamicGRU(nn.Module):
      def __init__(self, encoder_embedding_input_dim, encoder_embedding_output_dim, enc_ouput_dim, n_encoders, decoder_embedding_input_dim, decoder_embedding_output_dim, dec_ouput_dim, n_decoders, linear_dim, dropout_rate):
            super(DynamicGRU, self).__init__()

            self.enc_embedding_vocab_size = encoder_embedding_input_dim
            self.enc_embedding_dim = encoder_embedding_output_dim
            self.enc_dim = enc_ouput_dim
            self.n_encoders = n_encoders

            self.dec_embedding_vocab_size = decoder_embedding_input_dim
            self.dec_embedding_dim = decoder_embedding_output_dim
            self.dec_dim = dec_ouput_dim
            self.n_decoders = n_decoders

            self.linear_dim = linear_dim

            self.encoder_embedding = nn.Embedding(self.enc_embedding_vocab_size, self.enc_embedding_dim)
            self.encoder = nn.GRU(self.enc_embedding_dim, self.enc_dim, num_layers = self.n_encoders, batch_first = True)

            self.decoder_embedding = nn.Embedding(self.dec_embedding_vocab_size, self.dec_embedding_dim)
            self.decoder = nn.GRU(self.dec_embedding_dim, self.dec_dim, num_layers = self.n_decoders, batch_first = True)

            self.fc1 = nn.Linear(self.dec_dim, self.linear_dim)
            self.fc2 = nn.Linear(self.linear_dim, self.dec_embedding_vocab_size)

            self.project_encoder_hidden_to_decoder_hidden = nn.Linear(self.n_encoders * self.enc_dim, self.n_decoders * self.dec_dim)
            # self.project_encoder_cell_to_decoder_cell = nn.Linear(self.n_encoders * self.enc_dim, self.n_decoders * self.dec_dim)

      def forward(self, x, y=None):
            x = self.encoder_embedding(x)
            output, enc_hidden_state = self.encoder(x)
            # note : hidden_state_dim == cell_state_dim
            if self.n_decoders == self.n_encoders:
                  dec_hidden = enc_hidden_state
            else:
                  """
                  Projecting the encoder hidden state to decoder hidden state dimensions 
                  to ensure size mismatch when n_decoders != n_encoders
                  
                  """
                  enc_hidden_state = enc_hidden_state.transpose(0, 1).contiguous()
                  enc_hidden_state = enc_hidden_state.view(enc_hidden_state.size(0), -1)

                  enc_hidden_state = self.project_encoder_hidden_to_decoder_hidden(enc_hidden_state)

                  dec_hidden = enc_hidden_state.view(enc_hidden_state.size(0), self.n_decoders, self.dec_dim)
                  dec_hidden = dec_hidden.transpose(0, 1).contiguous()
            
            dec_input = torch.tensor([[2]] * x.size(0), device = 'mps')
            outputs = []
            for i in range(10):
                  dec_embed = self.decoder_embedding(dec_input)
                  dec_output, dec_hidden = self.decoder(dec_embed, dec_hidden)
                  output = self.fc1(dec_output.squeeze(1))
                  output = self.fc2(output)
                  outputs.append(output)
                  if y is not None:
                        dec_input = y[:,i]
                  else:
                        dec_input = output.argmax(-1).unsqueeze(1)
            return torch.stack(outputs, 1)

      
            





            

           