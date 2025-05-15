from data import NativeLatinDataset
from torch.utils.data import DataLoader
from model import RNN, DynamicRNN, DynamicLSTM, DynamicGRU
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from configure import Configure
from argument_parser import parser

def train_model(model, epochs, train_loader, criterion, optimizer, device):

            for epoch in range(epochs):
                  model.train()
                  total_loss = 0
                  for x,y in tqdm(train_loader):
                        native = x
                        latin = y
                        native, latin = native.to(device), latin.to(device)
                        optimizer.zero_grad()
                        output = model(latin) 
                        # print(output.shape)
                        # break
                        output_dim = output.shape[-1]
                        output = output[:, 0:].reshape(-1, output_dim)  # exclude SOS
                        # print(output.shape)
                        native = native[:, 0:].reshape(-1)
                        
                        # print(native.shape)
                        loss = criterion(output, native)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # gradient clipping
                        optimizer.step()
                        total_loss += loss.item()
                  print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
                  evaluate_model(model, val_loader=val_dl, latinidx2char = train_dataset.latin_idx2char,nativeidx2char = train_dataset.native_idx2char, criterion = criterion, device = device)

def evaluate_model(model, val_loader, latinidx2char, nativeidx2char, criterion, device='cpu', max_len=20):
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
            # print(output.shape)
            # logits = outputs  # assuming model returns raw logits

            # Flatten for loss: (batch * max_len, vocab_size)
            # logits_flat = logits.view(-1, logits.size(-1))
            # targets_flat = native.view(-1)

            # loss = criterion(logits_flat, targets_flat)
            output_dim = output.shape[-1]
            output = output[:, 0:].reshape(-1, output_dim)
            native = native[:, 0:].reshape(-1)
            loss = criterion(output, native)
            # total_loss += loss.item() * native.size(0)
            total_loss += loss.item()

            preds = output.argmax(-1)  # (batch, max_len)
            char_matches += (preds == native).sum().item()
            char_total += preds.size(0)
            # print(output.shape, preds.shape, native.shape)
            # print(target.shape, pred.shape)
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
            input_str = ''.join([latinidx2char.get(idx, '?') for idx in inp if idx not in [0,1,3,4]])
            pred_str = ''.join([nativeidx2char.get(idx, '?') for idx in pred if idx not in [0,1,3,4]])
            tgt_str = ''.join([nativeidx2char.get(idx, '?') for idx in tgt if idx not in [0,1,3,4]])
            print(f"[{i+1}] Input: {input_str}\n    Pred : {pred_str}\n    Truth: {tgt_str}\n")

      # avg_loss = total_loss / total
      acc = correct / total * 100
      char_acc = char_matches / char_total * 100
      print(f"Total correct:{correct} Total samples : {total}")
      print(f"\nValidation Loss: {total_loss/len(val_dl):.4f}")
      print(f"Validation Accuracy (exact match [Word Based]): {acc:.2f}%\n")
      print(f"Validation Accuracy (exact match [char Based]): {char_acc:.2f}%\n")

            # Store sample predictions
            

    

    

    

    return total_loss, acc

if __name__ == '__main__':

      # TRAIN_PATH = "../dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv"
      # VAL_PATH = "../dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv"
      # TEST_PATH = "../dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv"
      device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

      args = parser.parse_args()

      config = args.__dict__

      configuration = Configure(config)

      train_dataset, val_dataset, test_dataset = configuration.get_datasets()

      train_dl, val_dl, test_dl = configuration.get_dataloders()

      model = configuration.get_model().to(device)


      # # train_dataset = NativeLatinDataset('hi')
      # # train_dl = DataLoader(train_dataset, batch_size = 256, shuffle = True)

      # # val_dataset = NativeLatinDataset('hi', 'dev')
      # # val_dl = DataLoader(val_dataset, batch_size = 256, shuffle = True)

      
      # # vx, vy = next(iter(val_dl))

      

      # # x, y = next(iter(train_dl))
      # # print(x.shape, y.shape)
      # # print(f"Length of the vocab : {len(train_dataset.native_char2idx)}")
      # # model = RNN(len(train_dataset.native_char2idx), len(train_dataset.latin_char2idx)).to(device)
      # model = DynamicGRU(len(train_dataset.latin_char2idx), 128, 128, 2, len(train_dataset.native_char2idx), 128, 128, 2, 256, 0).to(device)

      criterion = nn.CrossEntropyLoss(ignore_index=0)
      optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

      

      train_model(model, 10,train_dl, criterion, optimizer, device)

      
      


      