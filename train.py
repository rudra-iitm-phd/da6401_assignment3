from data import NativeLatinDataset
from torch.utils.data import DataLoader
from model import RNN, DynamicRNN, DynamicLSTM
import torch
import torch.nn as nn
from tqdm.auto import tqdm

if __name__ == '__main__':

      TRAIN_PATH = "../dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv"
      VAL_PATH = "../dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv"
      TEST_PATH = "../dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv"
      device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


      train_dataset = NativeLatinDataset('bn')
      dataloader = DataLoader(train_dataset, batch_size = 256, shuffle = True)
      x, y = next(iter(dataloader))
      print(x.shape, y.shape)
      print(f"Length of the vocab : {len(train_dataset.native_char2idx)}")
      # model = RNN(len(train_dataset.native_char2idx), len(train_dataset.latin_char2idx)).to(device)
      model = DynamicLSTM(len(train_dataset.latin_char2idx), 128, 128, 2, len(train_dataset.native_char2idx), 128, 128, 2, 256, 0).to(device)

      criterion = nn.CrossEntropyLoss(ignore_index=0)
      optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

      for epoch in range(10):
            model.train()
            total_loss = 0
            for x,y in tqdm(dataloader):
                  native = x
                  latin = y
                  native, latin = native.to(device), latin.to(device)
                  optimizer.zero_grad()
                  output = model(latin) 
                  # print(output.shape)
                  # break
                  output_dim = output.shape[-1]
                  output = output[:, 1:].reshape(-1, output_dim)  # exclude SOS
                  native = native[:, 1:].reshape(-1)
                  loss = criterion(output, native)
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # gradient clipping
                  optimizer.step()
                  total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
      


      