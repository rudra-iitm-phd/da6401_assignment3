
# 🌍🔤 Transliteration with Seq2Seq + Attention ⚡

> 🚀 A robust and configurable character-level transliteration model built using PyTorch, featuring RNN/LSTM/GRU backbones, optional attention, beam search, and connectivity visualization. Developed as part of **DA6401: Advanced Machine Learning** at IIT Madras.

---

## 🧠 Key Features

- 🔁 **Encoder-Decoder Architecture** (RNN / LSTM / GRU)
- 🧲 **Attention Mechanism** (Bahdanau-style additive attention)
- 🧪 **Supports Beam Search** decoding
- 📊 **Hyperparameter Sweeps** with Weights & Biases
- 📈 **Interactive Attention Visualization** with HTML & JS
- 🌐 **Supports Multiple Indian Languages** (Dakshina dataset)
- 🎯 **Modular Design** – easy to plug and play components

---

## 🗂️ Project Structure

```bash
📦 da6401_assignment3/
│
├── 🧠 Core Model
│   ├── model.py              # Dynamic Seq2Seq model with optional attention
│   └── shared.py             # Global vocab mappings (char2idx, idx2char)
│
├── 🧾 Data & Preprocessing
│   ├── data.py               # Dataset class for loading & encoding Dakshina data
│   └── configure.py          # Loads dataset, builds model and optimizer
│
├── 🏋️‍♂️ Training & Evaluation
│   ├── train.py              # Training loop, evaluation, logging, visualization
│   ├── best_configs.py       # Predefined high-performing hyperparameter sets
│   ├── sweep_configuration.py# W&B hyperparameter sweep configurations
│   └── argument_parser.py    # Command-line arguments for configuration
│
├── 🌐 Visualization
│   └── connectivity_html.py  # HTML-based attention visualization with Devanagari support
│
├── 📂 Output & Logs
│   ├── predictions_vanilla/  # Stores predictions from the vanilla model
│   ├── wandb/                # W&B run logs and metadata
│   └── __pycache__/          # Python bytecode cache (auto-generated)
│
├── 📄 README.md              # Project overview and documentation
└── 📄 .DS_Store              # (Optional) macOS system metadata file

## 🏃‍♂️ Quickstart

### 1. 🧹 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 📦 Train a Model

```bash
python train.py -n hi -m lstm -ua True --wandb
```

> Use `--help` to see all CLI options.

### 3. 🧪 Run W&B Sweep

```bash
wandb sweep sweep_configuration.py
wandb agent <your-sweep-id>
```

---

## 🖼️ Attention Connectivity Visualization

> 🧠 Gain insights into model behavior with interactive HTML-based attention maps.

✅ Shows attention links between characters  
✅ Highlights top-3 predicted tokens with probabilities  
✅ Supports Devanagari font rendering  

---

## 🧠 Model Architecture

```
Input (Latin)
   ↓
Embedding → Encoder (RNN/LSTM/GRU) ──► Hidden States
                                          ↓
                              Attention Context (optional)
                                          ↓
Decoder (with or without attention) → Output (Native Script)
```

---

## 📊 Example Hyperparameters (Best Configs)

### 🔥 Attention Model
```python
{
  "activation": "relu",
  "batch_size": 128,
  "dropout_rate": 0.6,
  "epochs": 20,
  "model": "lstm",
  ...
}
```

---

## 📚 Dataset

🗂️ [Dakshina Dataset v1.0](https://github.com/google-research-datasets/dakshina)  
📁 Expected path: `../dakshina_dataset_v1.0/<lang>/lexicons/<lang>.translit.sampled.{train,dev,test}.tsv`

---

## 🔧 CLI Arguments & Configuration Options

Here are all the command-line options you can use with `train.py`, based on `argument_parser.py`:

| 🏷️ **Flag** | 🔑 **Argument** | 💬 **Description** | 🧠 **Type** | 🎯 **Default** |
|-------------|------------------|--------------------|------------|----------------|
| `-b`        | `--batch_size`      | 📦 Batch size | `int`     | `128`          |
| `-n`        | `--native`          | 🌐 Native language code (e.g. `hi`, `bn`, `ta`, etc.) | `str` | `'hi'` |
| `-m`        | `--model`           | 🧠 Model type: `rnn`, `lstm`, `gru` | `str` | `'lstm'` |
| `-eed`      | `--enc_embedding_dim` | 🧱 Encoder embedding size | `int` | `64` |
| `-ed`       | `--enc_dim`         | 🔢 Encoder hidden size | `int` | `512` |
| `-ne`       | `--n_encoders`      | 📚 Number of encoder layers | `int` | `1` |
| `-ded`      | `--dec_embedding_dim` | 🧱 Decoder embedding size | `int` | `64` |
| `-dd`       | `--dec_dim`         | 🔢 Decoder hidden size | `int` | `512` |
| `-nd`       | `--n_decoders`      | 📚 Number of decoder layers | `int` | `1` |
| `-ld`       | `--linear_dim`      | 📐 Hidden size of the linear layer | `int` | `2048` |
| `-do`       | `--dropout_rate`    | 💧 Dropout probability | `float` | `0.4` |
| `-e`        | `--epochs`          | ⏳ Number of training epochs | `int` | `20` |
| `-ua`       | `--use_attn`        | 🧲 Enable attention mechanism | `bool` | `False` |
| `-lc`       | `--log_connectivity`| 🖼️ Log HTML attention visualization | `bool` | `False` |
| `-mom`      | `--momentum`        | 🌀 Optimizer momentum (for Adam family) | `float` | `0.935939775` |
| `-k`        | `--beam_size`       | 🌬️ Beam width for beam search | `int` | `3` |
| `-a`        | `--activation`      | ⚙️ Activation function: `relu`, `tanh`, `gelu` | `str` | `'relu'` |
| `-o`        | `--optimizer`       | 🧪 Optimizer: `adam`, `adamw`, `adamax`, `rmsprop` | `str` | `'adamw'` |
| `-lr`       | `--learning_rate`   | 🧮 Learning rate | `float` | `0.00319` |
| `--use_v2`  |                    | 🧪 Use alternate attention configuration (v2) | `bool` | `False` |
| `--use_test`|                    | 🧪 Evaluate on test set after training | `bool` | `False` |
| `--wandb`   |                    | 📊 Enable Weights & Biases logging | `bool` | `False` |
| `-we`       | `--wandb_entity`    | 🧑‍🔬 W&B entity name | `str` | `'da24d008-iit-madras'` |
| `-wp`       | `--wandb_project`   | 📁 W&B project name | `str` | `'da6401-assignment3'` |
| `--wandb_sweep` |               | 🔄 Enable W&B sweep mode | `bool` | `False` |
| `--sweep_id`|                    | 🆔 W&B sweep ID to resume | `str` | `None` |

---

## 🧑‍💻 Author

👤 Developed by Rudra Sarkar, DA24D008, PhD , Data science and Artificial Intelligence Dept, IIT Madras  
📬 Feel free to reach out for academic collaborations or extensions!

---

## 🛡️ License

This project is licensed for academic and educational use.  
If adapting, please **cite or attribute** appropriately.

---

## ⭐ Acknowledgements

- 🤗 [PyTorch](https://pytorch.org/)
- 🧪 [Weights & Biases](https://wandb.ai/)
- 🇮🇳 [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)

---

### 🌟 If you find this useful, consider leaving a ⭐ on GitHub!
