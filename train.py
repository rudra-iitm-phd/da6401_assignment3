from data import NativeLatinDataset
from torch.utils.data import DataLoader
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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import matplotlib as mpl
import io, requests
from matplotlib import font_manager as fm, pyplot as plt
import tempfile
import csv
import best_configs
import matplotlib
matplotlib.use('Agg')

# from PIL import Image
# from matplotlib.colors import LinearSegmentedColormap
# import base64
import warnings
from connectivity_html import create_connectivity_html
# Suppress UserWarnings from matplotlib and seaborn

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning, module="wandb")

url = (
    "https://github.com/googlefonts/noto-fonts/"
    "raw/main/hinted/ttf/NotoSansDevanagari/"
    "NotoSansDevanagari-Regular.ttf"
)
r = requests.get(url); r.raise_for_status()

tmpdir = tempfile.gettempdir()
font_path = os.path.join(tmpdir, "NotoSansDevanagari-Regular.ttf")
with open(font_path, "wb") as f:
    f.write(r.content)

fm.fontManager.addfont(font_path)
hindi_prop = fm.FontProperties(fname=font_path)

def log_connectivity_visualization_to_wandb(model, input_seq, target_seq, latinidx2char, nativeidx2char, device):
    """
    Generate and log connectivity visualization to wandb with top-3 predictions
    
    Args:
        model: The sequence-to-sequence model with attention
        input_seq: Input sequence (tensor)
        target_seq: Target sequence (tensor)
        latinidx2char: Dictionary mapping from latin indices to characters
        nativeidx2char: Dictionary mapping from native indices to characters
        device: Device to run model on ('cpu', 'cuda', etc.)
    """
    model.eval()
    
    with torch.no_grad():
        # Get model output and attention weights
        output, attn_weights = model(input_seq.unsqueeze(0).to(device))
        
        # Convert indices to characters
        input_tokens = [latinidx2char.get(idx.item(), '?') for idx in input_seq if idx.item() not in [0, 1, 2, 3]]
        
        # Get predicted probabilities and tokens
        output_probs = torch.softmax(output, dim=-1)[0]  # Get probs for batch item 0
        
        # Get top-3 predictions for each position
        top3_predictions = []
        for pos in range(output_probs.size(0)):
            if pos < len(output_probs):
                probs, indices = output_probs[pos].topk(3)
                top3 = [(nativeidx2char.get(idx.item(), '?'), prob.item()) 
                       for idx, prob in zip(indices, probs)]
                top3_predictions.append(top3)
        
        # Get the actual predicted tokens (top-1)
        pred_indices = output.argmax(-1)[0]
        pred_tokens = [nativeidx2char.get(idx.item(), '?') for idx in pred_indices if idx.item() not in [0, 1, 2, 3]]
        
        # Create HTML visualization with predictions
        html_content = create_connectivity_html(
            input_tokens, 
            pred_tokens, 
            config["model"], 
            attn_weights[0],
            top3_predictions
        )
        
        # Log to wandb
        wandb.log({
            "connectivity_visualization": wandb.Html(html_content)
        })

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
                        if config['use_attn']:
                              output, _ = model(latin, native)
                        else:
                              output = model(latin, native)
                        teacher_forcing = teacher_forcing / max(0.8 * t, 0.1)
                        t = t + 1
                  else:
                        if config['use_attn']:
                              output, attn_weights = model(latin) 
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

            val_loss, val_char_acc, val_word_acc = evaluate_model(model, data_loader=val_loader, latinidx2char = shared.latin_idx2char,nativeidx2char = shared.native_idx2char, criterion = criterion, device = device, beam_size = beam_size, data = 'val', log = True if log else False, log_connectivity = config['log_connectivity'])
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

def plot_attention(attn_matrix, input_tokens, output_tokens):
      fig, ax = plt.subplots(figsize=(8,6))

      heat = sns.heatmap(attn_matrix.cpu().detach().numpy(),
                  xticklabels=True,
                  yticklabels=True,
                  cmap='viridis', cbar=True, ax=ax)
      cbar = heat.collections[0].colorbar
      cbar.set_label("Attention weight", fontdict={'size':12})
      # set ticks
      ax.set_xticks(range(len(input_tokens)))
      ax.set_yticks(range(len(output_tokens)))
      # English x-axis: default font
      ax.set_xticklabels(input_tokens, rotation=90)

      # Hindi y-axis: force Devanagari font
      ax.set_yticklabels(output_tokens,
                        rotation=0,
                        fontproperties=hindi_prop)
      ax.set_xlabel('Input tokens')
      ax.set_ylabel('Output tokens')
      # plt.tight_layout()
      return fig

def evaluate_model(model, data_loader, latinidx2char, nativeidx2char, criterion, device='cpu', data='val', beam_size = 1, log = False, log_connectivity = False):
      model.eval()
      total_loss = 0
      total = 0
      correct = 0
      samples = []
      char_matches = 0
      char_total = 0

      all_preds = []
      all_targets = []

      if data == 'test':
            input_latin = []
            pred_native = []

      with torch.no_grad():
            for native, latin in data_loader:

                  # native, latin = batch
                  native = native.to(device)
                  latin = latin.to(device)

                  target = native
                  
                  if config['use_attn']:
                        
                        output, attn_weights = model(latin)  # shape: (batch, max_len, vocab_size)
                  else:
                        output = model(latin)
                  
                  pred = output.argmax(-1)

                  output_dim = output.shape[-1]
                  output = output[:, 0:].reshape(-1, output_dim)
                  native = native[:, 0:].reshape(-1)
                  loss = criterion(output, native)
                  total_loss += loss.item()

                  inp_tok, pred_tok = [], []

                  if beam_size == 1:

                        """ Char level accuracy """
                        preds = output.argmax(-1) # (batch, max_len)
                        pred_mask = preds.ne(3) & preds.ne(1) & preds.ne(2) & preds.ne(0)
                        native_mask = native.ne(3) & native.ne(1) & native.ne(2) & native.ne(0)
                        mask = pred_mask & native_mask
                        char_matches += (preds[mask] == native[mask]).sum().item()
                        char_total += mask.sum().item()
                        
                        
                        # inp_token = [latinidx2char.get(idx.item(), '?') for idx in latin[0] ]
                        # inp_tok.append(inp_token)
                        # pred_token = [nativeidx2char.get(idx.item(), '?') for idx in pred[0]]
                        # pred_tok.append(pred_token)

                  
         

                  elif beam_size > 1:
                        """ Char level accuracy """
                        pred, attn_weights = model.beam(latin, k = beam_size)
                        preds = pred.reshape(-1)
                        native = native.reshape(-1)
                        pred_mask = preds.ne(3) & preds.ne(1) & preds.ne(2) & preds.ne(0)
                        native_mask = native.ne(3) & native.ne(1) & native.ne(2) & native.ne(0)
                        mask = pred_mask & native_mask
                        char_matches += (preds[mask] == native[mask]).sum().item()
                        char_total += mask.sum().item()

                  inp_token = [latinidx2char.get(idx.item(), '?') for idx in latin[0] ]
                  inp_tok.append(inp_token)
                  pred_token = [nativeidx2char.get(idx.item(), '?') for idx in pred[0]]
                  pred_tok.append(pred_token)

                  if data == 'test':
                              for (x, y) in zip(latin, pred):
                                    x_token = [latinidx2char.get(idx.item(), '?') for idx in x if idx.item() not in [0,1,2,3]]
                                    y_token = [nativeidx2char.get(idx.item(), '?') for idx in y if idx.item() not in [0,1,2,3]]
                                    
                                    input_latin.append(''.join(i for i in x_token))
                                    pred_native.append(''.join(i for i in y_token))


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

      attn_sample = attn_weights[0] if config['use_attn'] else None

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
                  if char_acc > 50.0 :

                        if config['use_attn'] :
                              fig = plot_attention(attn_sample, inp_tok[0], pred_tok[0])
                                    
                              plt.close(fig)

                              wandb.log(
                                          {'attention_heatmap': wandb.Image(fig)}
                                          
                                    )
                        if data == 'test':
                              if config['use_attn']:
                                    wandb.log(
                                          {
                                                "Test Accuracy with Attention (word)":acc,
                                                "Test Accuracy with Attention (char)":char_acc
                                          }
                                    )
                              else:
                                    wandb.log(
                                          {
                                                "Test Accuracy (word)":acc,
                                                "Test Accuracy (char)":char_acc
                                          }
                                    )
                        if log_connectivity and data == 'val' and config['use_attn']:
                              log_connectivity_visualization_to_wandb(
                                                                        model, 
                                                                        latin[0], 
                                                                        native[0], 
                                                                        latinidx2char, 
                                                                        nativeidx2char, 
                                                                        device
                                                                        )
                        

      if data == 'test':
            return total_loss/len(data_loader), char_acc, acc, input_latin, pred_native
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

      if args.use_test:
            if config['use_attn']:
                  config.update(best_configs.attn)
            else:
                  config.update(best_configs.vanilla)

            loss, char_acc, word_acc, test_inp, test_pred = evaluate_model(trained_model, test_dl, shared.latin_idx2char, shared.native_idx2char, criterion, device, 'test', 1, log, config['log_connectivity'])
            if config['use_attn']:
                  generate_csv(test_inp, test_pred, 'predictions_attention/predictions_attention.csv')
            else:
                  generate_csv(test_inp, test_pred, 'predictions_vanilla/predictions_vanilla.csv')

def generate_csv(test_inp, test_pred, output_path = None):
      output_path = "predictions.csv" if None else output_path
      with open(output_path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Latin Word', 'Predicted Native Word'])
            for latin, pred in zip(test_inp, test_pred):
                  writer.writerow([latin, pred])


if __name__ == '__main__':

      device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

      args = parser.parse_args()

      config = args.__dict__
      if args.wandb_sweep:
            if args.use_attn :
                  if args.use_v2 :
                        sweep_config = sweep_configuration.sweep_config_with_attn_v2
                  else:
                        sweep_config = sweep_configuration.sweep_config_with_attn
            else:
                  sweep_config = sweep_configuration.sweep_config
            if not args.sweep_id:
                  sweep_id = wandb.sweep(sweep_config, project=config['wandb_project'], entity=config['wandb_entity'])
            else:
                  sweep_id = args.sweep_id

            wandb.agent(sweep_id, function=train_core, count=100)
            wandb.finish()
      if args.wandb:
            train_core(log=True, sweep=False)
      else:
            train_core(log=False, sweep=False)

      

      