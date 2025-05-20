# Sweep configuration for training without attention
sweep_config = {
    "method": "bayes",  # Use Bayesian optimization to sample hyperparameters
    "metric": {"name": "Accuracy(char)", "goal": "maximize"},  # Metric to optimize

    "parameters": {
        # Activation function to use in model
        "activation": {"values": ['relu', 'gelu', 'tanh']},

        # Batch size for training
        "batch_size": {"values": [128, 256, 512, 1024]},

        # Encoder embedding dimensions
        "enc_embedding_dim": {"values": [64, 128, 256, 512]},
        # Encoder hidden state dimension
        "enc_dim": {"values": [64, 128, 256, 512]},
        # Number of encoder layers
        "n_encoders": {"values": [1, 2, 3, 4]},

        # Decoder embedding dimensions
        "dec_embedding_dim": {"values": [64, 128, 256, 512]},
        # Decoder hidden state dimension
        "dec_dim": {"values": [64, 128, 256, 512]},
        # Number of decoder layers
        "n_decoders": {"values": [1, 2, 3, 4]},

        # Dimension of the linear layer between decoder and final output
        "linear_dim": {"values": [64, 128, 256, 512, 1024, 2048]},

        # Dropout rate applied throughout the network
        "dropout_rate": {"values": [0, 0.2, 0.4, 0.6]},

        # Beam search sizes for decoding
        "beam_size": {"values": [1, 2, 3]},

        # Optimizer choices for training
        "optimizer": {"values": ['adam', 'adamw', 'adamax', 'rmsprop']},

        # Learning rate (log-uniform distribution)
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},

        # Momentum factor (used for optimizers that support it)
        "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.99},

        # RNN model type
        "model": {"values": ["lstm", "gru", "rnn"]}

        # Optional: weight decay (currently commented out)
        # "weight_decay": {"distribution": "uniform", "min": 0, "max": 1e-2},
    }
}

# Sweep configuration for training WITH attention
sweep_config_with_attn = {
    "method": "bayes",  # Bayesian optimization for hyperparams
    "metric": {"name": "Accuracy(char)", "goal": "maximize"},  # Optimize char-level accuracy

    "parameters": {
        "activation": {"values": ['relu', 'gelu', 'tanh']},
        "batch_size": {"values": [128, 256, 512, 1024]},

        "enc_embedding_dim": {"values": [64, 128, 256, 512]},
        "enc_dim": {"values": [64, 128, 256, 512]},
        "n_encoders": {"values": [1, 2]},  # Reduced for attention models

        "dec_embedding_dim": {"values": [64, 128, 256, 512]},
        "dec_dim": {"values": [64, 128, 256, 512]},
        "n_decoders": {"values": [1, 2]},  # Reduced for attention

        "linear_dim": {"values": [64, 128, 256, 512, 1024, 2048]},
        "dropout_rate": {"values": [0, 0.2, 0.4, 0.6]},
        "beam_size": {"values": [1]},  # Fixed beam size in this config

        "optimizer": {"values": ['adam', 'adamw', 'adamax']},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.99},

        # Explicitly turn on attention
        "use_attn": {"values": [True]},
        "model": {"values": ["lstm", "gru", "rnn"]}
    }
}

# Sweep configuration for attention v2 (more decoder depth/options)
sweep_config_with_attn_v2 = {
    "method": "bayes",
    "metric": {"name": "Accuracy(char)", "goal": "maximize"},

    "parameters": {
        "activation": {"values": ['relu', 'gelu', 'tanh']},
        "batch_size": {"values": [128, 256, 512, 1024]},

        "enc_embedding_dim": {"values": [64, 128, 256, 512]},
        "enc_dim": {"values": [64, 128, 256, 512]},
        "n_encoders": {"values": [1, 2, 3]},  # More encoder layers allowed

        "dec_embedding_dim": {"values": [64, 128, 256, 512]},
        "dec_dim": {"values": [64, 128, 256, 512]},
        "n_decoders": {"values": [1, 2, 3]},  # More decoder layers allowed

        "linear_dim": {"values": [64, 128, 256, 512, 1024, 2048]},
        "dropout_rate": {"values": [0, 0.2, 0.4, 0.6]},
        "beam_size": {"values": [1, 2, 3]},

        "optimizer": {"values": ['adam', 'adamw', 'adamax']},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.99},

        "use_attn": {"values": [True]},
        "model": {"values": ["lstm", "gru", "rnn"]}
    }
}
