from data import NativeLatinDataset
from torch.utils.data import DataLoader
# from model import DynamicSeq2Seq
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from configure import Configure
from argument_parser import parser
import numpy as np
import torch.nn.init as init
import wandb
import shared
import sweep_configuration

# Generate a readable name for the W&B run
def create_name(configuration:dict):
      l = [f'{k}-{v}' for k,v in configuration.items() if k not in ['wandb_entity', 'wandb_project', 'wandb_sweep', 'sweep_id', 'wandb']]
      return '_'.join(l)

def train_model(model, epochs, train_loader, val_loader, criterion, optimizer, device, beam_size=1, log=False):

      teacher_forcing = 0.7
      t = 1
      correct, total = 0, 0   
      char_matches, char_total = 0, 0
      
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

                  # pred = output.argmax(-1)
                  # target = native

                  output_dim = output.shape[-1]
                  output = output[:, 0:].reshape(-1, output_dim) 
                  native = native[:, 0:].reshape(-1)

                  loss = criterion(output, native)
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # gradient clipping
                  optimizer.step()
                  total_loss += loss.item()


            train_loss = total_loss/len(train_loader)
            # _, train_char_acc, train_word_acc = evaluate_model(model, data_loader=train_loader, latinidx2char = shared.latin_idx2char,nativeidx2char = shared.native_idx2char, criterion = criterion, device = device, beam_size = 1, data = 'train')

            val_loss, val_char_acc, val_word_acc = evaluate_model(model, data_loader=val_loader, latinidx2char = shared.latin_idx2char,nativeidx2char = shared.native_idx2char, criterion = criterion, device = device, beam_size = beam_size, data = 'val', log = True if log else False)
            if log:
                  wandb.log({
                        "Accuracy(char)":val_char_acc,
                        "Accuracy(word)":val_word_acc,
                        "Val loss":val_loss,
                        # "Train Accuracy(char)":train_char_acc,
                        # "Train Accuracy(word)":train_word_acc,
                        "Train loss":train_loss
                  })
      return model

def compute_confusion_matrix(preds, targets, num_classes):
      cm = np.zeros((num_classes, num_classes), dtype=int)
      for t, p in zip(targets, preds):
            cm[t][p] += 1
      return cm

def evaluate_model(model, data_loader, latinidx2char, nativeidx2char, criterion, device='cpu', data='val', beam_size = 1, log = False):
      model.eval()
      total_loss = 0
      total = 0
      correct = 0
      samples = []
      char_matches = 0
      char_total = 0

      all_preds = []
      all_targets = []

      with torch.no_grad():
            for native, latin in data_loader:
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

                  if beam_size == 1:

                        """ Char level accuracy """
                        preds = output.argmax(-1) # (batch, max_len)
                        pred_mask = preds.ne(3) & preds.ne(1) & preds.ne(2) & preds.ne(0)
                        native_mask = native.ne(3) & native.ne(1) & native.ne(2) & native.ne(0)
                        mask = pred_mask & native_mask
                        char_matches += (preds[mask] == native[mask]).sum().item()
                        char_total += mask.sum().item()
                        

                  elif beam_size > 1:
                        """ Char level accuracy """
                        pred = model.beam(latin, k = beam_size)
                        preds = pred.reshape(-1)
                        native = native.reshape(-1)

                        pred_mask = preds.ne(3) & preds.ne(1) & preds.ne(2) & preds.ne(0)
                        native_mask = native.ne(3) & native.ne(1) & native.ne(2) & native.ne(0)
                        mask = pred_mask & native_mask
                        char_matches += (preds[mask] == native[mask]).sum().item()
                        char_total += mask.sum().item()

                  y_pred = preds[mask].detach().tolist()
                  y_true = native[mask].detach().tolist()

                  all_preds += y_pred
                  all_targets += y_true

                  cm = compute_confusion_matrix(all_preds, all_targets, num_classes=len(nativeidx2char))

                  """ Word level accuracy """
                  pred[pred == 3] = 0 # masking out the padding and eos tags
                  target[target == 3] = 0

                  matches = (pred == target).all(-1)
                  correct += matches.sum().item()
                  total += pred.size(0)

            sample_table = []
            if data in ['test' , 'val']:
                  
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
                        sample_table.append([input_str, pred_str, tgt_str])
                        print(f"[{i+1}] Input: {input_str}\n    Pred : {pred_str}\n    Truth: {tgt_str}\n")

            if log:
                  wandb.log({
                        "confusion_matrix": wandb.plot.confusion_matrix(
                              probs=None,
                              y_true=all_targets,
                              preds=all_preds,
                              class_names=[nativeidx2char[i] for i in range(len(nativeidx2char))]
                        ), 
                        "sample predictions":wandb.Table(columns=["Input", "Prediction", "Target"], data = sample_table)
                        })


      acc = correct / total * 100
      char_acc = char_matches / char_total * 100
      print(f"Total correct:{correct} Total samples : {total}")
      if data=='test':
            print(f"Test Accuracy (exact match [Word Based]): {acc:.2f}%\n")
            print(f"Test Accuracy (exact match [char Based]): {char_acc:.2f}%\n")
      elif data == 'val':
            print(f"\nValidation Loss: {total_loss/len(data_loader):.4f}")
            print(f"Validation Accuracy (exact match [Word Based]): {acc:.2f}%\n")
            print(f"Validation Accuracy (exact match [char Based]): {char_acc:.2f}%\n")
      elif data == 'train':
            print(f"\nTrain Loss: {total_loss/len(data_loader):.4f}")
            print(f"Train Accuracy (exact match [Word Based]): {acc:.2f}%\n")
            print(f"Train Accuracy (exact match [char Based]): {char_acc:.2f}%\n")

      return total_loss/len(data_loader), char_acc, acc

def train_core(log=True, sweep=True):
      if log:
            # Initialize W&B run
            run =  wandb.init(entity = config['wandb_entity'], project = config['wandb_project'], config = config)

            if log and sweep:
                  sweep_config = wandb.config
                  config.update(sweep_config)

            # Give a name to this W&B run based on config
            run.name = create_name(wandb.config)

      configuration = Configure(config)

      train_dataset, val_dataset, test_dataset = configuration.get_datasets()

      train_dl, val_dl, test_dl = configuration.get_dataloders()

      model = configuration.get_model().to(device)
      
      criterion = nn.CrossEntropyLoss(ignore_index=0)
      # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
      optimizer = configuration.get_optimzer()
   
      if log:
            trained_model = train_model(model, config['epochs'], train_dl, val_dl, criterion, optimizer, device, beam_size = config['beam_size'], log = True)
      else:
            trained_model = train_model(model, config['epochs'], train_dl, val_dl, criterion, optimizer, device, beam_size = config['beam_size'], log = False)


if __name__ == '__main__':

      device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

      args = parser.parse_args()

      config = args.__dict__
      if args.wandb_sweep:
            sweep_config = sweep_configuration.sweep_config
            if not args.sweep_id:
                  sweep_id = wandb.sweep(sweep_config, project=config['wandb_project'], entity=config['wandb_entity'])
            else:
                  sweep_id = args.sweep_id

            wandb.agent(sweep_id, function=train_core, count=300)
            wandb.finish()
      if args.wandb:
            train_core(log=True, sweep=False)
      else:
            train_core(log=False, sweep=False)

      

      