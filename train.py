from data import NativeLatinDataset
from torch.utils.data import DataLoader
from model import RNN, DynamicRNN, DynamicLSTM
import torch
import torch.nn as nn
from tqdm.auto import tqdm

def evaluate_model(model, val_loader, latinidx2char, nativeidx2char, criterion, device='cpu', max_len=20):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    samples = []

    with torch.no_grad():
        for batch in val_loader:
            native, latin = batch
            native = native.to(device)
            latin = latin.to(device)


            outputs = model(latin)  # shape: (batch, max_len, vocab_size)
            logits = outputs  # assuming model returns raw logits

            # Flatten for loss: (batch * max_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = native.view(-1)

            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item() * latin.size(0)

            preds = logits.argmax(-1)  # (batch, max_len)
            matches = (preds == native).all(dim=1)
            correct += matches.sum().item()
            total += native.size(0)

            # Store sample predictions
            if len(samples) < 10:
                for i in range(min(10 - len(samples), latin.size(0))):
                    input_seq = latin[i].tolist()
                    pred_seq = preds[i].tolist()
                    target_seq = native[i].tolist()
                    samples.append((input_seq, pred_seq, target_seq))

    avg_loss = total_loss / total
    acc = correct / total * 100

    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy (exact match): {acc:.2f}%\n")

    print("Sample Predictions:")
    for i, (inp, pred, tgt) in enumerate(samples):
        input_str = ''.join([latinidx2char.get(idx, '?') for idx in inp if idx not in [0,1,3,4]])
        pred_str = ''.join([nativeidx2char.get(idx, '?') for idx in pred if idx not in [0,1,3,4]])
        tgt_str = ''.join([nativeidx2char.get(idx, '?') for idx in tgt if idx not in [0,1,3,4]])
        print(f"[{i+1}] Input: {input_str}\n    Pred : {pred_str}\n    Truth: {tgt_str}\n")

    return avg_loss, acc

if __name__ == '__main__':

      TRAIN_PATH = "../dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv"
      VAL_PATH = "../dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv"
      TEST_PATH = "../dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv"
      device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


      train_dataset = NativeLatinDataset('bn')
      dataloader = DataLoader(train_dataset, batch_size = 256, shuffle = True)

      val_dataset = NativeLatinDataset('bn', 'dev')
      val_dl = DataLoader(val_dataset, batch_size = 256, shuffle = True)

      
      vx, vy = next(iter(val_dl))

      

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

            evaluate_model(model, val_loader=val_dl, latinidx2char = train_dataset.latin_idx2char,nativeidx2char = train_dataset.native_idx2char, criterion = criterion, device = device)

      
      


      