import torch 
import torch.nn as nn

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


            

           