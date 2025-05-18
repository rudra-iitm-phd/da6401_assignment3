import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class EncoderDecoderAttention(nn.Module):
      def __init__(self, dec_dim):
            super(EncoderDecoderAttention, self).__init__()

            self.dec_dim = dec_dim 
            self.W_enc = nn.Linear(self.dec_dim, self.dec_dim)
            self.W_dec = nn.Linear(self.dec_dim, self.dec_dim)
            self.v = nn.Parameter(torch.rand(self.dec_dim))
            self.tanh = nn.Tanh()

      def forward(self, dec_hidden, enc_output, mask=None):
            batch_size, seq_len, _ = enc_output.size() # batch x seq_len x enc_dim
            Uhj = self.W_dec(dec_hidden) # o/p --> batch x hidden
            Uhj = Uhj.unsqueeze(1) # o/p --> batch x 1 x hidden
            Ws = self.W_enc(enc_output) # o/p --> batch x seq_len x hidden 
            energy = self.tanh(Uhj+Ws) # batch x seq_len x hidden
            v = self.v.unsqueeze(0).unsqueeze(2) # 1 x hidden x 1
            v = v.repeat(batch_size, 1, 1)
            scores = torch.bmm(energy, v).squeeze(-1) # batch x seq_len
            if mask is not None:
                  scores = scores.masked_fill(mask==0, -1e9)
            attn_weights = F.softmax(scores, dim = 1) # batch x seq_len
            attn_weights = attn_weights.unsqueeze(1) # batch x 1 x seq_len
            context = torch.bmm(attn_weights, enc_output) # batch x 1 x enc_dim
            # context = context.squeeze(1) # batch x enc_dim
            return context, attn_weights
            


class DynamicSeq2Seq(nn.Module):
      def __init__(self, model_type, activation, encoder_embedding_input_dim, encoder_embedding_output_dim, enc_ouput_dim, n_encoders, decoder_embedding_input_dim, decoder_embedding_output_dim, dec_ouput_dim, n_decoders, linear_dim, dropout_rate, use_attn):
            super(DynamicSeq2Seq, self).__init__()

            self.enc_embedding_vocab_size = encoder_embedding_input_dim
            self.enc_embedding_dim = encoder_embedding_output_dim
            self.enc_dim = enc_ouput_dim
            self.n_encoders = n_encoders

            self.dec_embedding_vocab_size = decoder_embedding_input_dim
            self.dec_embedding_dim = decoder_embedding_output_dim
            self.dec_dim = dec_ouput_dim
            self.n_decoders = n_decoders
            self.dropout_rate = dropout_rate

            self.use_attention = use_attn

            self.linear_dim = linear_dim

            self.model_map = {
                  "rnn":nn.RNN,
                  "lstm":nn.LSTM,
                  "gru":nn.GRU
            }

            self.activation_map = {
                  "relu":nn.ReLU,
                  "tanh":nn.Tanh,
                  "gelu":nn.GELU
            }

            self.activation = activation
            
            self.model_type = model_type.lower()
            self.model = self.model_map[self.model_type]

            self.encoder_embedding = nn.Embedding(self.enc_embedding_vocab_size, self.enc_embedding_dim)
            self.encoder = self.model(self.enc_embedding_dim, self.enc_dim, num_layers = self.n_encoders, batch_first = True, dropout = dropout_rate)
            

            self.decoder_embedding = nn.Embedding(self.dec_embedding_vocab_size, self.dec_embedding_dim)
            if self.use_attention :
                  self.decoder = self.model(self.dec_embedding_dim + self.dec_dim, self.dec_dim, num_layers = self.n_decoders, batch_first = True, dropout = dropout_rate)
            else :
                  self.decoder = self.model(self.dec_embedding_dim, self.dec_dim, num_layers = self.n_decoders, batch_first = True, dropout = dropout_rate)
            
            

            self.fc1 = nn.Linear(self.dec_dim, self.linear_dim)
            self.activation = self.activation_map[activation.lower()]()
            
            if self.dropout_rate > 0:
                  self.linear_dropout = nn.Dropout(p = self.dropout_rate)
            self.fc2 = nn.Linear(self.linear_dim, self.dec_embedding_vocab_size)

            self.project_encoder_hidden_to_decoder_hidden = nn.Linear(self.n_encoders * self.enc_dim, self.n_decoders * self.dec_dim)
            self.project_encoder_cell_to_decoder_cell = nn.Linear(self.n_encoders * self.enc_dim, self.n_decoders * self.dec_dim)

            if use_attn:
                  self.attention = EncoderDecoderAttention(self.dec_dim)
                  self.project_attention_to_dec_hidden = nn.Linear(2*self.dec_dim , self.dec_dim)
                  if self.enc_dim != self.dec_dim:
                        self.project_encoder_output_decoder_hidden = nn.Linear(self.enc_dim, self.dec_dim)

      def forward(self, x, y=None):
            device = x.device
            x = self.encoder_embedding(x)

            if self.model_type == 'lstm':
                  enc_output, (enc_hidden_state, enc_cell_state) = self.encoder(x)
            else:
                  enc_output, enc_hidden_state = self.encoder(x)
            # note : hidden_state_dim == cell_state_dim
            if self.n_decoders == self.n_encoders and self.enc_dim == self.dec_dim:
                  dec_hidden = enc_hidden_state

                  if self.model_type == 'lstm':
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

                  if self.model_type == 'lstm':
                        """
                        Projecting the encoder cell state to decoder cell state dimensions 
                        to ensure size mismatch when n_decoders != n_encoders
                        
                        """
                        enc_cell_state = enc_cell_state.transpose(0, 1).contiguous()
                        enc_cell_state = enc_cell_state.view(enc_cell_state.size(0), -1)

                        enc_cell_state = self.project_encoder_cell_to_decoder_cell(enc_cell_state)

                        dec_cell = enc_cell_state.view(enc_cell_state.size(0), self.n_decoders, self.dec_dim)
                        dec_cell = dec_cell.transpose(0, 1).contiguous()

            
            dec_input = torch.tensor([[2]] * x.size(0), device = device)

            if self.use_attention:
                        attention_weights = torch.zeros(x.size(0), x.size(1), x.size(1))
                        if self.enc_dim != self.dec_dim:
                              enc_output = enc_output.reshape(-1, self.enc_dim)
                              enc_output = self.project_encoder_output_decoder_hidden(enc_output)
                              enc_output = enc_output.reshape(x.size(0), x.size(1), -1)
            
            outputs = []
            for i in range(10):
                  dec_embed = self.decoder_embedding(dec_input)
                  if self.use_attention:
                        context, attn_weights = self.attention(dec_hidden[-1], enc_output)
                        attention_weights[:,i,:] = attn_weights.squeeze(1)
                        dec_embed = torch.cat([dec_embed, context], dim = -1)
                        context = context.squeeze(1).repeat(self.n_decoders, 1, 1)
                        # dec_hidden = nn.Tanh()(self.project_attention_to_dec_hidden(torch.cat([dec_hidden, context], dim = -1)))
                        # dec_hidden = self.project_attention_to_dec_hidden(torch.cat([dec_hidden, context], dim = -1))
                        dec_hidden = context
                        
                  if self.model_type == 'lstm':
                        # dec_hidden = context if self.use_attention else dec_hidden
                        # hidden_cell, attn_weights = self.attention(dec_cell[-1], enc_output)
                        # dec_cell = hidden_cell.repeat(self.n_decoders, 1, 1) if self.use_attention else dec_hidden
                        dec_output, (dec_hidden, dec_cell) = self.decoder(dec_embed, (dec_hidden, dec_cell))
                  else:
                        dec_output, dec_hidden = self.decoder(dec_embed, dec_hidden)
                  output = self.fc1(dec_output.squeeze(1))
                  output = self.activation(output)
                  if self.dropout_rate > 0:
                        output = self.linear_dropout(output)
                  output = self.fc2(output)
                  outputs.append(output)
                  if y is not None:
                        dec_input = y[:,i].view(-1, 1)
                  else:
                        dec_input = output.argmax(-1).unsqueeze(1)
            return torch.stack(outputs, 1), attention_weights

      def beam(self, x, k: int = 2, max_length: int = 10):
            batch_size = x.size(0)
            device = x.device

            x = self.encoder_embedding(x)
            if self.model_type == 'lstm':
                  _, (enc_h, enc_c) = self.encoder(x)
            else:
                  _, enc_h = self.encoder(x)

            if self.n_decoders == self.n_encoders and self.enc_dim == self.dec_dim:
                  if self.model_type == 'lstm':
                        dec_h, dec_c = enc_h, enc_c
                  else:
                        dec_h = enc_h

            else:
                  # project encoder → decoder dims
                  enc_h = enc_h.transpose(0, 1).reshape(batch_size, -1)
                  dec_h = self.project_encoder_hidden_to_decoder_hidden(enc_h)\
                              .view(batch_size, self.n_decoders, self.dec_dim)\
                              .transpose(0, 1).contiguous()

                  if self.model_type == 'lstm':
                        enc_c = enc_c.transpose(0, 1).reshape(batch_size, -1)
                        dec_c = self.project_encoder_cell_to_decoder_cell(enc_c)\
                                    .view(batch_size, self.n_decoders, self.dec_dim)\
                                    .transpose(0, 1).contiguous()

            start_tok = 2
            seqs   = torch.full((batch_size, k, 1), start_tok, dtype=torch.long, device=device)  # [B,k,1]
            scores = torch.zeros(batch_size, k, device=device)                                   # [B,k]

            dec_h = dec_h.repeat_interleave(k, dim=1)  # [n_layers, B*k, dim]
            dec_c = dec_c.repeat_interleave(k, dim=1) if self.model_type == 'lstm' else None

            vocab = self.dec_embedding_vocab_size
            for t in range(max_length):
                  last_tok = seqs[:, :, -1].reshape(batch_size * k, 1)
                  if self.model_type == 'lstm':
                        dec_out, (dec_h, dec_c) = self.decoder(
                              self.decoder_embedding(last_tok), (dec_h, dec_c)
                        )
                  else:
                        dec_out, dec_h = self.decoder(self.decoder_embedding(last_tok), dec_h)
                  
                  logits = self.fc1(dec_out.squeeze(1))
                  logits = self.activation(logits)
                  logits = self.linear_dropout(logits) if self.dropout_rate > 0 else logits
                  logits = self.fc2(logits)
                  logp   = F.log_softmax(logits, dim=-1).view(batch_size, k, vocab)  # [B,k,V]

                  # top-k per beam (returns [B,k,k] for both)
                  tk_logp, tk_tok = logp.topk(k, dim=-1)

                  # Flatten the (beam,token) grid → (beam*token) and add old scores
                  new_scores = (scores.unsqueeze(2) + tk_logp).view(batch_size, -1)  # [B,k*k]
                  top_scores, flat_idx = new_scores.topk(k, dim=-1)                  # best k overall

                  # Convert flat index back to (beam_row, token_col) without gather
                  beam_row = flat_idx.div(k, rounding_mode='floor')   # integer division
                  tok_col  = flat_idx.remainder(k)

                  # Get next tokens via advanced indexing (no gather)
                  next_tok = tk_tok[torch.arange(batch_size).unsqueeze(1),
                                    beam_row, tok_col]                # [B,k]

                  # Update sequences with pure indexing
                  seqs = seqs[torch.arange(batch_size).unsqueeze(1), beam_row]       # choose beams
                  seqs = torch.cat([seqs, next_tok.unsqueeze(-1)], dim=-1)           # append step

                  scores = top_scores                                                # update scores

            best = scores.argmax(dim=1)
            best_seq = seqs[torch.arange(batch_size), best]  # [B, max_length+1]
            return best_seq[:, 1:max_length+1]

      
            





            

           