import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F  # For softmax and other activation utilities

# Attention mechanism for encoder-decoder architecture
class EncoderDecoderAttention(nn.Module):
      def __init__(self, dec_dim):
            super(EncoderDecoderAttention, self).__init__()

            self.dec_dim = dec_dim  # Dimension of decoder hidden state (and attention space)
            self.W_enc = nn.Linear(self.dec_dim, self.dec_dim)  # Linear layer for encoder outputs
            self.W_dec = nn.Linear(self.dec_dim, self.dec_dim)  # Linear layer for decoder hidden
            self.v = nn.Parameter(torch.rand(self.dec_dim))  # Learnable vector for scoring attention
            self.tanh = nn.Tanh()  # Activation for energy scoring

      def forward(self, dec_hidden, enc_output, mask=None):
            # dec_hidden: [batch, dec_dim]
            # enc_output: [batch, seq_len, enc_dim]

            batch_size, seq_len, _ = enc_output.size()
            Uhj = self.W_dec(dec_hidden)  # [batch, dec_dim]
            Uhj = Uhj.unsqueeze(1)  # Expand to [batch, 1, dec_dim] for broadcasting
            Ws = self.W_enc(enc_output)  # [batch, seq_len, dec_dim]
            energy = self.tanh(Uhj + Ws)  # [batch, seq_len, dec_dim]
            v = self.v.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)  # [batch, dec_dim, 1]
            scores = torch.bmm(energy, v).squeeze(-1)  # [batch, seq_len] - attention scores
            if mask is not None:
                  scores = scores.masked_fill(mask == 0, -1e9)  # Apply mask for padded positions
            attn_weights = F.softmax(scores, dim=1)  # Normalize over encoder steps
            attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, seq_len]
            context = torch.bmm(attn_weights, enc_output)  # [batch, 1, enc_dim]
            return context, attn_weights  # Return weighted context vector and attention weights


# Dynamic sequence-to-sequence model supporting RNN, LSTM, GRU and attention
class DynamicSeq2Seq(nn.Module):
      def __init__(self, model_type, activation, encoder_embedding_input_dim, encoder_embedding_output_dim, enc_ouput_dim, n_encoders, decoder_embedding_input_dim, decoder_embedding_output_dim, dec_ouput_dim, n_decoders, linear_dim, dropout_rate, use_attn):
            super(DynamicSeq2Seq, self).__init__()

            # Encoder configuration
            self.enc_embedding_vocab_size = encoder_embedding_input_dim
            self.enc_embedding_dim = encoder_embedding_output_dim
            self.enc_dim = enc_ouput_dim
            self.n_encoders = n_encoders

            # Decoder configuration
            self.dec_embedding_vocab_size = decoder_embedding_input_dim
            self.dec_embedding_dim = decoder_embedding_output_dim
            self.dec_dim = dec_ouput_dim
            self.n_decoders = n_decoders

            self.dropout_rate = dropout_rate
            self.use_attention = use_attn
            self.linear_dim = linear_dim  # Intermediate linear layer dimension

            # Map strings to actual RNN module classes
            self.model_map = {
                  "rnn": nn.RNN,
                  "lstm": nn.LSTM,
                  "gru": nn.GRU
            }

            # Map string names to activation functions
            self.activation_map = {
                  "relu": nn.ReLU,
                  "tanh": nn.Tanh,
                  "gelu": nn.GELU
            }

            self.activation = activation  # Activation function name
            self.model_type = model_type.lower()
            self.model = self.model_map[self.model_type]  # RNN/LSTM/GRU module

            # Encoder embedding + RNN stack
            self.encoder_embedding = nn.Embedding(self.enc_embedding_vocab_size, self.enc_embedding_dim)
            self.encoder = self.model(self.enc_embedding_dim, self.enc_dim, num_layers=self.n_encoders, batch_first=True, dropout=dropout_rate)

            # Decoder embedding
            self.decoder_embedding = nn.Embedding(self.dec_embedding_vocab_size, self.dec_embedding_dim)

            # Decoder RNN optionally with attention (input is context + embedded)
            if self.use_attention:
                  self.decoder = self.model(self.dec_embedding_dim + self.dec_dim, self.dec_dim, num_layers=self.n_decoders, batch_first=True, dropout=dropout_rate)
            else:
                  self.decoder = self.model(self.dec_embedding_dim, self.dec_dim, num_layers=self.n_decoders, batch_first=True, dropout=dropout_rate)

            # Projection from decoder hidden to output vocabulary space
            self.fc1 = nn.Linear(self.dec_dim, self.linear_dim)
            self.activation = self.activation_map[activation.lower()]()
            if self.dropout_rate > 0:
                  self.linear_dropout = nn.Dropout(p=self.dropout_rate)
            self.fc2 = nn.Linear(self.linear_dim, self.dec_embedding_vocab_size)

            # Linear projection to align encoder and decoder dimensions if needed
            self.project_encoder_hidden_to_decoder_hidden = nn.Linear(self.n_encoders * self.enc_dim, self.n_decoders * self.dec_dim)
            self.project_encoder_cell_to_decoder_cell = nn.Linear(self.n_encoders * self.enc_dim, self.n_decoders * self.dec_dim)

            # Attention mechanism + projection if encoder/decoder dims mismatch
            if use_attn:
                  self.attention = EncoderDecoderAttention(self.dec_dim)
                  if self.enc_dim != self.dec_dim:
                        self.project_encoder_output_decoder_hidden = nn.Linear(self.enc_dim, self.dec_dim)

      def forward(self, x, y=None):
            device = x.device
            x = self.encoder_embedding(x)  # [batch, seq_len] -> [batch, seq_len, embed_dim]

            # Encoder pass
            if self.model_type == 'lstm':
                  enc_output, (enc_hidden_state, enc_cell_state) = self.encoder(x)
            else:
                  enc_output, enc_hidden_state = self.encoder(x)

            # Align encoder hidden to decoder hidden state
            if self.n_decoders == self.n_encoders and self.enc_dim == self.dec_dim:
                  dec_hidden = enc_hidden_state
                  if self.model_type == 'lstm':
                        dec_cell = enc_cell_state
            else:
                  # Project encoder hidden state
                  enc_hidden_state = enc_hidden_state.transpose(0, 1).contiguous().view(enc_hidden_state.size(0), -1)
                  enc_hidden_state = self.project_encoder_hidden_to_decoder_hidden(enc_hidden_state)
                  dec_hidden = enc_hidden_state.view(enc_hidden_state.size(0), self.n_decoders, self.dec_dim).transpose(0, 1).contiguous()

                  if self.model_type == 'lstm':
                        enc_cell_state = enc_cell_state.transpose(0, 1).contiguous().view(enc_cell_state.size(0), -1)
                        enc_cell_state = self.project_encoder_cell_to_decoder_cell(enc_cell_state)
                        dec_cell = enc_cell_state.view(enc_cell_state.size(0), self.n_decoders, self.dec_dim).transpose(0, 1).contiguous()

            dec_input = torch.tensor([[2]] * x.size(0), device=device)  # Start token <sos>

            if self.use_attention:
                  attention_weights = torch.zeros(x.size(0), x.size(1), x.size(1))  # [batch, dec_len, enc_len]
                  if self.enc_dim != self.dec_dim:
                        enc_output = self.project_encoder_output_decoder_hidden(enc_output.view(-1, self.enc_dim)).view(x.size(0), x.size(1), -1)

            outputs = []
            for i in range(10):  # Decoder runs for fixed length of 10
                  dec_embed = self.decoder_embedding(dec_input)  # [batch, 1, embed_dim]

                  if self.use_attention:
                        context, attn_weights = self.attention(dec_hidden[-1], enc_output)
                        attention_weights[:, i, :] = attn_weights.squeeze(1)  # Store attention weights
                        dec_embed = torch.cat([dec_embed, context], dim=-1)  # Add context vector
                        context = context.squeeze(1).repeat(self.n_decoders, 1, 1)
                        dec_hidden = context + dec_hidden  # Residual attention update

                  if self.model_type == 'lstm':
                        dec_output, (dec_hidden, dec_cell) = self.decoder(dec_embed, (dec_hidden, dec_cell))
                  else:
                        dec_output, dec_hidden = self.decoder(dec_embed, dec_hidden)

                  output = self.fc1(dec_output.squeeze(1))  # [batch, dec_dim] -> [batch, linear_dim]
                  output = self.activation(output)
                  if self.dropout_rate > 0:
                        output = self.linear_dropout(output)
                  output = self.fc2(output)  # [batch, vocab_size]
                  outputs.append(output)

                  if y is not None:
                        dec_input = y[:, i].view(-1, 1)  # Teacher-forcing
                  else:
                        dec_input = output.argmax(-1).unsqueeze(1)  # Greedy decode

            if self.use_attention:
                  return torch.stack(outputs, 1), attention_weights
            else:
                  return torch.stack(outputs, 1)

      def beam(self, x, k: int = 2, max_length: int = 10):
            # Beam search decoding
            batch_size = x.size(0)
            device = x.device

            x = self.encoder_embedding(x)
            if self.model_type == 'lstm':
                  enc_output, (enc_h, enc_c) = self.encoder(x)
            else:
                  enc_output, enc_h = self.encoder(x)

            if self.n_decoders == self.n_encoders and self.enc_dim == self.dec_dim:
                  if self.model_type == 'lstm':
                        dec_h, dec_c = enc_h, enc_c
                  else:
                        dec_h = enc_h
            else:
                  enc_h = enc_h.transpose(0, 1).reshape(batch_size, -1)
                  dec_h = self.project_encoder_hidden_to_decoder_hidden(enc_h).view(batch_size, self.n_decoders, self.dec_dim).transpose(0, 1).contiguous()
                  if self.model_type == 'lstm':
                        enc_c = enc_c.transpose(0, 1).reshape(batch_size, -1)
                        dec_c = self.project_encoder_cell_to_decoder_cell(enc_c).view(batch_size, self.n_decoders, self.dec_dim).transpose(0, 1).contiguous()

            start_tok = 2  # <sos> token

            if self.use_attention:
                  attention_weights = torch.zeros(batch_size * k, max_length, x.size(1), device=device)
                  if self.enc_dim != self.dec_dim:
                        enc_output = self.project_encoder_output_decoder_hidden(enc_output.view(-1, self.enc_dim)).view(batch_size, x.size(1), -1)

            seqs = torch.full((batch_size, k, 1), start_tok, dtype=torch.long, device=device)  # Initial sequence
            scores = torch.zeros(batch_size, k, device=device)

            enc_output = enc_output.unsqueeze(1).repeat(1, k, 1, 1).view(batch_size * k, x.size(1), -1)
            dec_h = dec_h.repeat_interleave(k, dim=1)
            dec_c = dec_c.repeat_interleave(k, dim=1) if self.model_type == 'lstm' else None

            vocab = self.dec_embedding_vocab_size

            for t in range(max_length):
                  last_tok = seqs[:, :, -1].reshape(batch_size * k, 1)
                  dec_input = self.decoder_embedding(last_tok)

                  if self.use_attention:
                        context, attn_weights = self.attention(dec_h[-1], enc_output)
                        attention_weights[:, t, :] = attn_weights.squeeze(1)
                        dec_input = torch.cat([dec_input, context], dim=-1)
                        context = context.squeeze(1).unsqueeze(0).repeat(self.n_decoders, 1, 1)
                        dec_h = dec_h + context

                  if self.model_type == 'lstm':
                        dec_out, (dec_h, dec_c) = self.decoder(dec_input, (dec_h, dec_c))
                  else:
                        dec_out, dec_h = self.decoder(dec_input, dec_h)

                  logits = self.fc1(dec_out.squeeze(1))
                  logits = self.activation(logits)
                  logits = self.linear_dropout(logits) if self.dropout_rate > 0 else logits
                  logits = self.fc2(logits)
                  logp = F.log_softmax(logits, dim=-1).view(batch_size, k, vocab)

                  tk_logp, tk_tok = logp.topk(k, dim=-1)  # Top-k tokens and their scores
                  new_scores = (scores.unsqueeze(2) + tk_logp).view(batch_size, -1)
                  top_scores, flat_idx = new_scores.topk(k, dim=-1)

                  beam_row = flat_idx.div(k, rounding_mode='floor')  # Parent sequence index
                  tok_col = flat_idx.remainder(k)  # Token index

                  next_tok = tk_tok[torch.arange(batch_size).unsqueeze(1), beam_row, tok_col]
                  seqs = seqs[torch.arange(batch_size).unsqueeze(1), beam_row]
                  seqs = torch.cat([seqs, next_tok.unsqueeze(-1)], dim=-1)

                  scores = top_scores

                  beam_idx = beam_row + (torch.arange(batch_size) * k).unsqueeze(1).to(device)
                  dec_h = dec_h[:, beam_idx.view(-1), :]
                  if self.model_type == 'lstm':
                        dec_c = dec_c[:, beam_idx.view(-1), :]

            best = scores.argmax(dim=1)
            best_seq = seqs[torch.arange(batch_size), best]
            if self.use_attention:
                  return best_seq[:, 1:max_length+1], attention_weights
            return best_seq[:, 1:max_length+1]
