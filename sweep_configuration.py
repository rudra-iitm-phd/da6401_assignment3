sweep_config = {
            "method": "bayes",  # Use Bayesian optimization for hyperparameter tuning
            "metric": {"name": "Accuracy(char)", "goal": "maximize"},
            "parameters": {
                  # List of activation functions 
                  "activation": {"values": ['relu','gelu','tanh']},
                  # Batch size options
                  "batch_size": {"values": [128, 256, 512, 1024]},

                  "enc_embedding_dim":{"values":[64, 128, 256, 512]},
                  "enc_dim":{"values":[64, 128, 256, 512]},
                  "n_encoders":{"values":[1, 2, 3, 4]},

                  "dec_embedding_dim":{"values":[64, 128, 256, 512]},
                  "dec_dim":{"values":[64, 128, 256, 512]},
                  "n_decoders":{"values":[1, 2, 3, 4]},

                  "linear_dim":{"values":[64, 128, 256, 512, 1024, 2048]},

                  "dropout_rate":{"values":[0, 0.2, 0.4, 0.6]},

                  "beam_size":{"values":[1, 2, 3]},

                  "optimizer": {"values": ['adam', 'adamw', 'adamax', 'rmsprop']},
                 
                  "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
                  
                  "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.99},

                  # "weight_decay": {"distribution": "uniform", "min": 0, "max": 1e-2},
            }
        }