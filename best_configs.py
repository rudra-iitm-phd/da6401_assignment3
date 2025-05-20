attn = {
      "activation":"relu",
      "batch_size":128,
      "beam_size":1,
      "dec_dim":128,
      "dec_embedding_dim":512,
      "dropout_rate":0.4,
      "enc_dim":512,
      "enc_embedding_dim":128,
      "epochs":20,
      "learning_rate":0.00603939542740778,
      "linear_dim":2048,
      "model":"lstm",
      "momentum":0.8721411550264451,
      "n_decoders":2,
      "n_encoders":1,
      "native":"hi",
      "optimizer":"adamax",
      "use_attn":True

}

vanilla  = {
      "activation":"tanh",
      "batch_size":128,
      "beam_size":2,
      "dec_dim":512,
      "dec_embedding_dim":64,
      "dropout_rate":0.2,
      "enc_dim":512,
      "enc_embedding_dim":512,
      "epochs":20,
      "learning_rate":0.0008679588865292194,
      "linear_dim":128,
      "log_connectivity":False,
      "model":"lstm",
      "momentum":0.9390345827949242,
      "n_decoders":4,
      "n_encoders":4,
      "native":"hi",
      "optimizer":"adam"
}