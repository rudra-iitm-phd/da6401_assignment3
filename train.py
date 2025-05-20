# Imports for data handling and model training
from data import NativeLatinDataset  # Custom dataset class
from torch.utils.data import DataLoader  # Batching support
import torch
import torch.nn as nn
from tqdm.auto import tqdm  # Progress bar for loops
from configure import Configure  # Configuration manager
from argument_parser import parser  # CLI argument parser
import numpy as np
import torch.nn.init as init
import wandb  # Weights & Biases logging
import shared  # Shared vocab mappings
import sweep_configuration  # Sweep setup for hyperparameter search
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Visualization
import matplotlib.font_manager as fm  # Font management
import os
import matplotlib as mpl
import io, requests
from matplotlib import font_manager as fm, pyplot as plt
import tempfile
import csv
import best_configs  # Preset best configs
import matplotlib
matplotlib.use('Agg')  # Disable interactive GUI backend

# Ignore matplotlib/seaborn/wandb user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning, module="wandb")

# Load Devanagari font from Google Fonts for rendering Hindi
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
hindi_prop = fm.FontProperties(fname=font_path)  # Font properties for Hindi

from connectivity_html import create_connectivity_html  # HTML attention viz tool

# Log connectivity visualization as HTML artifact in wandb
def log_connectivity_visualization_to_wandb(model, input_seq, target_seq, latinidx2char, nativeidx2char, device):
    model.eval()  # Switch to eval mode
    
    with torch.no_grad():
        # Get output and attention weights from model
        output, attn_weights = model(input_seq.unsqueeze(0).to(device))
        
        # Convert Latin input indices to chars (excluding special tokens)
        input_tokens = [latinidx2char.get(idx.item(), '?') for idx in input_seq if idx.item() not in [0, 1, 2, 3]]
        
        # Compute softmax probabilities for each output step
        output_probs = torch.softmax(output, dim=-1)[0]
        
        # Top-3 predictions per output position
        top3_predictions = []
        for pos in range(output_probs.size(0)):
            if pos < len(output_probs):
                probs, indices = output_probs[pos].topk(3)
                top3 = [(nativeidx2char.get(idx.item(), '?'), prob.item()) for idx, prob in zip(indices, probs)]
                top3_predictions.append(top3)
        
        # Convert predicted indices to chars
        pred_indices = output.argmax(-1)[0]
        pred_tokens = [nativeidx2char.get(idx.item(), '?') for idx in pred_indices if idx.item() not in [0, 1, 2, 3]]
        
        # Generate HTML visualization
        html_content = create_connectivity_html(
            input_tokens, 
            pred_tokens, 
            config["model"], 
            attn_weights[0],
            top3_predictions
        )
        
        # Log HTML to wandb
        wandb.log({
            "connectivity_visualization": wandb.Html(html_content)
        })

# Generate readable name string for wandb runs
def create_name(configuration:dict):
      l = [f'{k}-{v}' for k,v in configuration.items() if k not in ['wandb_entity', 'wandb_project', 'wandb_sweep', 'sweep_id', 'wandb']]
      return '_'.join(l)

# Main training loop
def train_model(model, epochs, train_loader, val_loader, criterion, optimizer, device, beam_size=1, log=False):
      teacher_forcing = 0.7  # Initial teacher forcing ratio
      t = 1  # Epoch scaling variable for reducing teacher forcing
      correct, total = 0, 0  # Word-level metrics
      char_matches, char_total = 0, 0  # Char-level metrics
      
      for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x,y in tqdm(train_loader):  # Loop over batches
                  native = x  # Ground truth
                  latin = y  # Input (Latin)
                  native, latin = native.to(device), latin.to(device)
                  optimizer.zero_grad()

                  # Apply teacher forcing with decaying probability
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

                  # Reshape for loss computation
                  output_dim = output.shape[-1]
                  output = output[:, 0:].reshape(-1, output_dim) 
                  native = native[:, 0:].reshape(-1)

                  loss = criterion(output, native)
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
                  optimizer.step()
                  total_loss += loss.item()

            # Compute epoch training loss
            train_loss = total_loss / len(train_loader)

            # Run evaluation on validation set
            val_loss, val_char_acc, val_word_acc = evaluate_model(
                  model, data_loader=val_loader, 
                  latinidx2char=shared.latin_idx2char,
                  nativeidx2char=shared.native_idx2char, 
                  criterion=criterion, device=device, 
                  beam_size=beam_size, log=True if log else False, 
                  log_connectivity=config['log_connectivity'])

            if log:
                  wandb.log({
                        "Accuracy(char)": val_char_acc,
                        "Accuracy(word)": val_word_acc,
                        "Val loss": val_loss,
                        "Train loss": train_loss
                  })
      return model

# Confusion matrix: true vs predicted char frequency
def compute_confusion_matrix(preds, targets, num_classes):
      cm = np.zeros((num_classes, num_classes), dtype=int)
      for t, p in zip(targets, preds):
            cm[t][p] += 1
      return cm

# Plot attention matrix using seaborn
def plot_attention(attn_matrix, input_tokens, output_tokens):
      fig, ax = plt.subplots(figsize=(8,6))
      heat = sns.heatmap(attn_matrix.cpu().detach().numpy(), xticklabels=True, yticklabels=True, cmap='viridis', cbar=True, ax=ax)
      cbar = heat.collections[0].colorbar
      cbar.set_label("Attention weight", fontdict={'size':12})
      ax.set_xticks(range(len(input_tokens)))
      ax.set_yticks(range(len(output_tokens)))
      ax.set_xticklabels(input_tokens, rotation=90)
      ax.set_yticklabels(output_tokens, rotation=0, fontproperties=hindi_prop)
      ax.set_xlabel('Input tokens')
      ax.set_ylabel('Output tokens')
      return fig

# Evaluate model performance
def evaluate_model(model, data_loader, latinidx2char, nativeidx2char, criterion, device='cpu', data='val', beam_size = 1, log = False, log_connectivity = False):
      model.eval()
      total_loss, total, correct = 0, 0, 0
      samples = []
      char_matches, char_total = 0, 0
      all_preds, all_targets = [], []

      if data == 'test':
            input_latin, pred_native = [], []

      with torch.no_grad():
            for native, latin in data_loader:
                  native = native.to(device)
                  latin = latin.to(device)
                  target = native

                  if config['use_attn']:
                        output, attn_weights = model(latin)
                  else:
                        output = model(latin)

                  pred = output.argmax(-1)

                  output_dim = output.shape[-1]
                  output = output[:, 0:].reshape(-1, output_dim)
                  native = native[:, 0:].reshape(-1)
                  loss = criterion(output, native)
                  total_loss += loss.item()

                  if beam_size == 1:
                        preds = output.argmax(-1)
                        pred_mask = preds.ne(3) & preds.ne(1) & preds.ne(2) & preds.ne(0)
                        native_mask = native.ne(3) & native.ne(1) & native.ne(2) & native.ne(0)
                        mask = pred_mask & native_mask
                        char_matches += (preds[mask] == native[mask]).sum().item()
                        char_total += mask.sum().item()

                  elif beam_size > 1:
                        if config['use_attn']:
                              pred, attn_weights = model.beam(latin, k=beam_size)
                        else:
                              pred = model.beam(latin, k=beam_size)
                        preds = pred.reshape(-1)
                        native = native.reshape(-1)
                        pred_mask = preds.ne(3) & preds.ne(1) & preds.ne(2) & preds.ne(0)
                        native_mask = native.ne(3) & native.ne(1) & native.ne(2) & native.ne(0)
                        mask = pred_mask & native_mask
                        char_matches += (preds[mask] == native[mask]).sum().item()
                        char_total += mask.sum().item()

                  attn_sample = attn_weights[0] if config['use_attn'] else None

                  if data == 'test':
                        inp_token = [latinidx2char.get(idx.item(), '?') for idx in latin[0]]
                        pred_token = [nativeidx2char.get(idx.item(), '?') for idx in pred[0]]
                        input_latin.append(''.join(i for i in inp_token if i not in ['<pad>', '<eos>', '<sos>', '<unk>']))
                        pred_native.append(''.join(i for i in pred_token if i not in ['<pad>', '<eos>', '<sos>', '<unk>']))
                        y_pred = preds[mask].detach().tolist()
                        y_true = native[mask].detach().tolist()
                        all_preds += y_pred
                        all_targets += y_true

                  pred[pred == 3] = 0
                  target[target == 3] = 0
                  matches = (pred == target).all(-1)
                  correct += matches.sum().item()
                  total += pred.size(0)

            acc = correct / total * 100
            char_acc = char_matches / char_total * 100

            if data == 'test' and log:
                  wandb.log({
                        "Test Accuracy (word)": acc,
                        "Test Accuracy (char)": char_acc
                  })

      if data == 'test':
            return total_loss / len(data_loader), char_acc, acc, input_latin, pred_native
      return total_loss / len(data_loader), char_acc, acc

# Main function to initialize training
def train_core(log=True, sweep=True):
      if args.use_test:
            if config['use_attn']:
                  config.update(best_configs.attn)
            else:
                  config.update(best_configs.vanilla)

      if log:
            run = wandb.init(entity=config['wandb_entity'], project=config['wandb_project'], config=config)
            if sweep:
                  sweep_config = wandb.config
                  config.update(sweep_config)
            run.name = create_name(wandb.config)

      configuration = Configure(config)
      train_dataset, val_dataset, test_dataset = configuration.get_datasets()
      train_dl, val_dl, test_dl = configuration.get_dataloders()
      model = configuration.get_model().to(device)

      criterion = nn.CrossEntropyLoss(ignore_index=0)
      optimizer = configuration.get_optimzer()

      trained_model = train_model(model, config['epochs'], train_dl, val_dl, criterion, optimizer, device, beam_size=config['beam_size'], log=log)

      if args.use_test:
            loss, char_acc, word_acc, test_inp, test_pred = evaluate_model(trained_model, test_dl, shared.latin_idx2char, shared.native_idx2char, criterion, device, 'test', 1, log, config['log_connectivity'])
            if config['use_attn']:
                  generate_csv(test_inp, test_pred, 'predictions_attention/predictions_attention.csv')
            else:
                  generate_csv(test_inp, test_pred, 'predictions_vanilla/predictions_vanilla.csv')

# Save predictions to CSV file
def generate_csv(test_inp, test_pred, output_path = None):
      output_path = "predictions.csv" if None else output_path
      with open(output_path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Latin Word', 'Predicted Native Word'])
            for latin, pred in zip(test_inp, test_pred):
                  writer.writerow([latin, pred])

# Entry point for script execution
if __name__ == '__main__':
      device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
      args = parser.parse_args()
      config = args.__dict__

      if args.wandb_sweep:
            if args.use_attn:
                  if args.use_v2:
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
