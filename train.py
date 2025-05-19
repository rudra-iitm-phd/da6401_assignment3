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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import matplotlib as mpl
import io, requests
from matplotlib import font_manager as fm, pyplot as plt
import tempfile
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import base64
import warnings
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

def create_connectivity_html(input_tokens, output_tokens, model_name, attn_weights, top3_predictions=None, threshold=0.1):
    """
    Create an HTML representation of the connectivity between input and output tokens
    with improved support for Devanagari script and top 3 token predictions on hover.
    
    Args:
        input_tokens: List of input characters
        output_tokens: List of output characters 
        attn_weights: Attention matrix of shape [output_len, input_len]
        top3_predictions: List of lists containing top 3 predictions for each output position
                         [[(token1, prob1), (token2, prob2), (token3, prob3)], ...]
        threshold: Minimum attention weight to show connection
        
    Returns:
        HTML string representation of the connectivity visualization
    """
    # Clean input and output tokens (remove padding, BOS, EOS)
    clean_input = [t for t in input_tokens if t not in ['<pad>', '<bos>', '<eos>', '?']]
    clean_output = [t for t in output_tokens if t not in ['<pad>', '<bos>', '<eos>', '?']]
    
    # Resize attention weights matrix to match cleaned tokens
    clean_attn = attn_weights[:len(clean_output), :len(clean_input)]
    
    # If no top3 predictions provided, create dummy data
    if top3_predictions is None:
        top3_predictions = [[("—", 0.0), ("—", 0.0), ("—", 0.0)] for _ in range(len(clean_output))]
    
    # Generate HTML for visualization
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&display=swap');
            
            body {{
                font-family: 'Arial', sans-serif;
            }}
            
            .container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                max-width: 900px;
                margin: 0 auto;
            }}
            
            .visualization {{
                display: flex;
                flex-direction: column;
                margin: 20px;
                position: relative;
                width: 100%;
            }}
            
            .row {{
                display: flex;
                justify-content: space-around;
                margin: 10px 0;
                position: relative;
                width: 100%;
            }}
            
            .input-row, .output-row {{
                display: flex;
                justify-content: space-around;
                width: 100%;
            }}
            
            .cell {{
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid #ddd;
                background-color: #f8f8f8;
                margin: 0 5px;
                position: relative;
                font-size: 18px;
                border-radius: 4px;
                cursor: pointer;
            }}
            
            .input-cell {{
                font-family: 'Arial', sans-serif;
            }}
            
            .output-cell {{
                font-family: 'Noto Sans Devanagari', sans-serif;
                font-size: 20px;
            }}
            
            .connection-container {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 0;
            }}
            
            .connection {{
                position: absolute;
                background-color: rgba(0, 128, 0, 0.5);
                height: 2px;
                transform-origin: 0 0;
                pointer-events: none;
                transition: opacity 0.2s, height 0.2s;
            }}
            
            .model-type {{
                font-weight: bold;
                font-size: 20px;
                margin: 10px;
                background-color: #eee;
                padding: 10px;
                border-radius: 5px;
            }}
            
            .highlighted {{
                background-color: #ffeb3b;
                box-shadow: 0 0 5px rgba(0,0,0,0.3);
                z-index: 10;
            }}
            
            .title {{
                font-weight: bold;
                font-size: 24px;
                margin: 20px;
            }}
            
            .instructions {{
                margin: 10px;
                font-style: italic;
                color: #666;
                text-align: center;
                max-width: 600px;
            }}
            
            .controls {{
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 15px 0;
            }}
            
            .slider-container {{
                display: flex;
                align-items: center;
                margin-right: 20px;
            }}
            
            .slider-label {{
                margin-right: 10px;
                font-weight: bold;
            }}
            
            #threshold-slider {{
                width: 150px;
            }}
            
            #threshold-value {{
                margin-left: 10px;
                min-width: 40px;
                text-align: center;
            }}
            
            /* Tooltip for top-3 predictions */
            .prediction-tooltip {{
                display: none;
                position: absolute;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                z-index: 100;
                width: 180px;
                top: 45px;
                left: 50%;
                transform: translateX(-50%);
                font-family: 'Noto Sans Devanagari', sans-serif;
            }}
            
            .prediction-tooltip::before {{
                content: '';
                position: absolute;
                top: -8px;
                left: 50%;
                transform: translateX(-50%);
                border-width: 0 8px 8px 8px;
                border-style: solid;
                border-color: transparent transparent #ddd transparent;
            }}
            
            .prediction-tooltip h4 {{
                margin: 0 0 8px 0;
                text-align: center;
                font-size: 14px;
                color: #444;
            }}
            
            .prediction-item {{
                display: flex;
                justify-content: space-between;
                margin: 4px 0;
                padding: 3px;
                border-radius: 3px;
            }}
            
            .prediction-item:hover {{
                background-color: #f5f5f5;
            }}
            
            .prediction-char {{
                font-size: 18px;
                font-weight: bold;
            }}
            
            .prediction-prob {{
                font-size: 14px;
                color: #666;
            }}
            
            .prediction-prob-bar {{
                height: 6px;
                background-color: #4caf50;
                margin-top: 2px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="title">Transliteration Attention Connectivity</div>
            <div class="instructions">
                Hover over any character to see its connections. 
                The green lines show the attention weights: stronger connections have darker green lines.
                Hover over output characters to see top prediction alternatives.
                Use the threshold slider to filter the connections by strength.
            </div>
            
            <div class="controls">
                <div class="slider-container">
                    <span class="slider-label">Threshold:</span>
                    <input type="range" id="threshold-slider" min="0" max="1" step="0.05" value="0.1">
                    <span id="threshold-value">0.1</span>
                </div>
            </div>
            
            <div class="model-type">Sequence-to-Sequence with Attention [ {} ]</div>
            <div class="visualization">
                <div class="row input-row">
    """.format(model_name.upper())
    
    # Add input tokens
    for i, token in enumerate(clean_input):
        html += f'<div class="cell input-cell" data-index="{i}" onmouseover="highlightInputConnections({i})" onmouseout="resetHighlights()">{token}</div>'
    
    html += """
                </div>
                <div class="connection-container">
    """
    
    # Add connections between input and output tokens
    for i, out_token in enumerate(clean_output):
        for j, in_token in enumerate(clean_input):
            weight = float(clean_attn[i, j].item())
            if weight > threshold:
                # Calculate opacity based on weight (normalized between 0.1 and 1.0)
                opacity = max(0.1, min(1.0, weight))
                
                # We'll calculate the position and rotation with JavaScript
                html += f'<div class="connection" data-from-input="{j}" data-to-output="{i}" data-weight="{weight:.4f}" style="opacity: 0;"></div>'
    
    html += """
                </div>
                <div class="row output-row">
    """
    
    # Add output tokens with tooltips for predictions
    for i, token in enumerate(clean_output):
        # Get top 3 predictions for this position
        top3 = top3_predictions[i] if i < len(top3_predictions) else [("—", 0), ("—", 0), ("—", 0)]
        
        # Create tooltip HTML for predictions
        tooltip_html = f"""
        <div class="prediction-tooltip" id="tooltip-{i}">
            <h4>Top Predictions</h4>
        """
        
        # Add each prediction with probability bar
        for idx, (pred_token, prob) in enumerate(top3):
            bar_width = int(prob * 100)
            is_actual = pred_token == token
            highlight = 'background-color: #e6f7e6;' if is_actual else ''
            
            tooltip_html += f"""
            <div class="prediction-item" style="{highlight}">
                <div class="prediction-char">{pred_token}</div>
                <div class="prediction-prob">
                    <div>{prob:.2f}</div>
                    <div class="prediction-prob-bar" style="width: {bar_width}%;"></div>
                </div>
            </div>
            """
        
        tooltip_html += "</div>"
        
        # Add output cell with tooltip
        html += f"""
        <div class="cell output-cell" data-index="{i}" 
             onmouseover="highlightOutputConnections({i}); showTooltip({i});" 
             onmouseout="resetHighlights(); hideTooltip({i});">
            {token}
            {tooltip_html}
        </div>
        """
    
    html += """
                </div>
            </div>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                positionConnections();
                window.addEventListener('resize', positionConnections);
                
                const thresholdSlider = document.getElementById('threshold-slider');
                const thresholdValue = document.getElementById('threshold-value');
                
                thresholdSlider.addEventListener('input', function() {
                    const value = parseFloat(this.value);
                    thresholdValue.textContent = value.toFixed(2);
                    updateConnectionVisibility(value);
                });
            });

            function updateConnectionVisibility(threshold) {
                const connections = document.querySelectorAll('.connection');
                connections.forEach(conn => {
                    const weight = parseFloat(conn.getAttribute('data-weight'));
                    conn.style.display = weight >= threshold ? 'block' : 'none';
                });
            }

            function positionConnections() {
                const connections = document.querySelectorAll('.connection');
                const inputCells = document.querySelectorAll('.input-cell');
                const outputCells = document.querySelectorAll('.output-cell');
                const containerRect = document.querySelector('.connection-container').getBoundingClientRect();
                
                connections.forEach(conn => {
                    const fromIndex = parseInt(conn.getAttribute('data-from-input'));
                    const toIndex = parseInt(conn.getAttribute('data-to-output'));
                    const weight = parseFloat(conn.getAttribute('data-weight'));
                    
                    if (fromIndex < inputCells.length && toIndex < outputCells.length) {
                        const fromCell = inputCells[fromIndex];
                        const toCell = outputCells[toIndex];
                        
                        const fromRect = fromCell.getBoundingClientRect();
                        const toRect = toCell.getBoundingClientRect();
                        
                        const fromX = fromRect.left + fromRect.width/2 - containerRect.left;
                        const fromY = fromRect.top + fromRect.height - containerRect.top;
                        const toX = toRect.left + toRect.width/2 - containerRect.left;
                        const toY = toRect.top - containerRect.top;
                        
                        const length = Math.sqrt(Math.pow(toX - fromX, 2) + Math.pow(toY - fromY, 2));
                        const angle = Math.atan2(toY - fromY, toX - fromX);
                        
                        conn.style.width = `${length}px`;
                        conn.style.left = `${fromX}px`;
                        conn.style.top = `${fromY}px`;
                        conn.style.transform = `rotate(${angle}rad)`;
                        
                        // Set opacity based on weight
                        conn.style.opacity = weight;
                        conn.style.backgroundColor = `rgba(0, 128, 0, ${weight})`;
                    }
                });
                
                // Apply initial threshold
                const thresholdSlider = document.getElementById('threshold-slider');
                if (thresholdSlider) {
                    updateConnectionVisibility(parseFloat(thresholdSlider.value));
                }
            }

            function highlightInputConnections(index) {
                const connections = document.querySelectorAll(`.connection[data-from-input="${index}"]`);
                const cell = document.querySelector(`.input-cell[data-index="${index}"]`);
                
                cell.classList.add('highlighted');
                
                connections.forEach(conn => {
                    const toIndex = conn.getAttribute('data-to-output');
                    document.querySelector(`.output-cell[data-index="${toIndex}"]`).classList.add('highlighted');
                    conn.style.height = '3px';
                    conn.style.zIndex = '5';
                });
            }

            function highlightOutputConnections(index) {
                const connections = document.querySelectorAll(`.connection[data-to-output="${index}"]`);
                const cell = document.querySelector(`.output-cell[data-index="${index}"]`);
                
                cell.classList.add('highlighted');
                
                connections.forEach(conn => {
                    const fromIndex = conn.getAttribute('data-from-input');
                    document.querySelector(`.input-cell[data-index="${fromIndex}"]`).classList.add('highlighted');
                    conn.style.height = '3px';
                    conn.style.zIndex = '5';
                });
            }

            function resetHighlights() {
                document.querySelectorAll('.highlighted').forEach(el => {
                    el.classList.remove('highlighted');
                });
                document.querySelectorAll('.connection').forEach(conn => {
                    conn.style.height = '2px';
                    conn.style.zIndex = '0';
                });
            }
            
            function showTooltip(index) {
                const tooltip = document.getElementById(`tooltip-${index}`);
                if (tooltip) {
                    tooltip.style.display = 'block';
                }
            }
            
            function hideTooltip(index) {
                const tooltip = document.getElementById(`tooltip-${index}`);
                if (tooltip) {
                    tooltip.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    
    return html

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
        
      #   # Also create a regular attention heatmap as a fallback
      #   fig, ax = plt.subplots(figsize=(10, 8))
      #   sns.heatmap(
      #       attn_weights[0].cpu().numpy()[:len(pred_tokens), :len(input_tokens)],
      #       xticklabels=input_tokens,
      #       yticklabels=pred_tokens,
      #       cmap='viridis',
      #       ax=ax
      #   )
      #   ax.set_xlabel('Input tokens (Latin)')
      #   ax.set_ylabel('Output tokens (Native)')
      #   ax.set_title('Attention Weights')
        
      #   # Log the regular heatmap as well
      #   wandb.log({"attention_heatmap": wandb.Image(fig)})
      #   plt.close(fig)

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

                  if beam_size == 1:

                        """ Char level accuracy """
                        preds = output.argmax(-1) # (batch, max_len)
                        pred_mask = preds.ne(3) & preds.ne(1) & preds.ne(2) & preds.ne(0)
                        native_mask = native.ne(3) & native.ne(1) & native.ne(2) & native.ne(0)
                        mask = pred_mask & native_mask
                        char_matches += (preds[mask] == native[mask]).sum().item()
                        char_total += mask.sum().item()
                        inp_tok, pred_tok = [], []
                        
                        inp_token = [latinidx2char.get(idx.item(), '?') for idx in latin[0] ]
                        inp_tok.append(inp_token)
                        pred_token = [nativeidx2char.get(idx.item(), '?') for idx in pred[0]]
                        pred_tok.append(pred_token)

                        attn_sample = attn_weights[0]
                        
                              

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
                        fig = plot_attention(attn_sample, inp_tok[0], pred_tok[0])
                              
                        plt.close(fig)

                        wandb.log(
                                    {'attention_heatmap': wandb.Image(fig)}
                                    
                              )
                        log_connectivity_visualization_to_wandb(
                                                                  model, 
                                                                  latin[0], 
                                                                  native[0], 
                                                                  latinidx2char, 
                                                                  nativeidx2char, 
                                                                  device
                                                                  )
                        


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
            if args.use_attn :
                  sweep_config = sweep_configuration.sweep_config_with_attn
            else:
                  sweep_config = sweep_configuration.sweep_config
            if not args.sweep_id:
                  sweep_id = wandb.sweep(sweep_config, project=config['wandb_project'], entity=config['wandb_entity'])
            else:
                  sweep_id = args.sweep_id

            wandb.agent(sweep_id, function=train_core, count=200)
            wandb.finish()
      if args.wandb:
            train_core(log=True, sweep=False)
      else:
            train_core(log=False, sweep=False)

      

      