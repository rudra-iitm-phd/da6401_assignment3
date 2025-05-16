from data import NativeLatinDataset
from torch.utils.data import DataLoader
from model import RNN, DynamicRNN, DynamicLSTM, DynamicGRU
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from configure import Configure
from argument_parser import parser
import numpy as np

def train_model(model, epochs, train_loader, criterion, optimizer, device):

      teacher_forcing = 0.7
      t = 1
      for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x,y in tqdm(train_loader):
                  native = x
                  latin = y
                  native, latin = native.to(device), latin.to(device)
                  optimizer.zero_grad()
                  if np.random.rand() <= teacher_forcing:
                        output = model(latin, native)
                        teacher_forcing = teacher_forcing / max(0.8 * t, 0.1)
                        t = t + 1
                  else:
                        output = model(latin) 
                  output_dim = output.shape[-1]
                  output = output[:, 0:].reshape(-1, output_dim) 
                  native = native[:, 0:].reshape(-1)
                  

                  loss = criterion(output, native)
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # gradient clipping
                  optimizer.step()
                  total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
            evaluate_model(model, val_loader=val_dl, latinidx2char = train_dataset.latin_idx2char,nativeidx2char = train_dataset.native_idx2char, criterion = criterion, device = device)
      return model

def evaluate_model(model, val_loader, latinidx2char, nativeidx2char, criterion, device='cpu', test=False):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    samples = []
    char_matches = 0
    char_total = 0


    with torch.no_grad():
      for native, latin in val_loader:
            # native, latin = batch
            native = native.to(device)
            latin = latin.to(device)

            target = native


            output = model(latin)  # shape: (batch, max_len, vocab_size)
            pred = output.argmax(-1)

            output_dim = output.shape[-1]
            output = output[:, 0:].reshape(-1, output_dim)
            native = native[:, 0:].reshape(-1)
            loss = criterion(output, native)
            total_loss += loss.item()

            """ Char level accuracy """
            preds = output.argmax(-1) # (batch, max_len)
            pred_mask = preds.ne(3) & preds.ne(1) & preds.ne(2) & preds.ne(0)
            native_mask = native.ne(3) & native.ne(1) & native.ne(2) & native.ne(0)
            mask = pred_mask & native_mask
            char_matches += (preds[mask] == native[mask]).sum().item()
            char_total += mask.sum().item()
            
            """ Word level accuracy """
            pred[pred == 3] = 0 # masking out the padding and eos tags
            target[target == 3] = 0

            matches = (pred == target).all(-1)
            correct += matches.sum().item()
            total += pred.size(0)

      if len(samples) < 5:
            for i in range(min(5 - len(samples), latin.size(0))):
                  input_seq = latin[i].tolist()
                  pred_seq = pred[i].tolist()
                  target_seq = target[i].tolist()
                  samples.append((input_seq, pred_seq, target_seq))

      print("Sample Predictions:")
      for i, (inp, pred, tgt) in enumerate(samples):
            input_str = ''.join([latinidx2char.get(idx, '?') for idx in inp if idx not in [0,1, 2, 3]])
            pred_str = ''.join([nativeidx2char.get(idx, '?') for idx in pred if idx not in [0,1, 2, 3]])
            tgt_str = ''.join([nativeidx2char.get(idx, '?') for idx in tgt if idx not in [0,1, 2, 3]])
            print(f"[{i+1}] Input: {input_str}\n    Pred : {pred_str}\n    Truth: {tgt_str}\n")


      acc = correct / total * 100
      char_acc = char_matches / char_total * 100
      print(f"Total correct:{correct} Total samples : {total}")
      if test:
            print(f"Test Accuracy (exact match [Word Based]): {acc:.2f}%\n")
            print(f"Test Accuracy (exact match [char Based]): {char_acc:.2f}%\n")
      else:
            print(f"\nValidation Loss: {total_loss/len(val_dl):.4f}")
            print(f"Validation Accuracy (exact match [Word Based]): {acc:.2f}%\n")
            print(f"Validation Accuracy (exact match [char Based]): {char_acc:.2f}%\n")

    return total_loss, acc

if __name__ == '__main__':

      device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

      args = parser.parse_args()

      config = args.__dict__

      configuration = Configure(config)

      train_dataset, val_dataset, test_dataset = configuration.get_datasets()

      train_dl, val_dl, test_dl = configuration.get_dataloders()

      model = configuration.get_model().to(device)

      criterion = nn.CrossEntropyLoss(ignore_index=0)
      optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
   

      trained_model = train_model(model, 2,train_dl, criterion, optimizer, device)
      evaluate_model(trained_model, test_dl, train_dataset.latin_idx2char, train_dataset.native_idx2char, criterion, device, test=True)

      
      


      