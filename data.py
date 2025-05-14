import torch 
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import shared




class NativeLatinDataset(Dataset):
      """
      Generate a torch dataset using the dakshina dataset

      """
      def __init__(self, lang, data='train'):
            
            
            self.path = f"../dakshina_dataset_v1.0/{lang}/lexicons/{lang}.translit.sampled.{data}.tsv"

            """ Get the native and latin vocabulary """
            self.native_words, self.latin_words = self.get_word_list(self.path)

            if data == 'train':
                  """ Mapping from character to index and index to character for native and latin vocabulary"""
                  self.native_char2idx = self.char2idx(self.native_words)
                  self.native_idx2char = {idx:char for (char,idx) in self.native_char2idx.items()}
                  shared.native_char2idx = self.native_char2idx
                  shared.native_idx2char = self.native_idx2char


                  self.latin_char2idx = self.char2idx(self.latin_words)
                  self.latin_idx2char = {idx:char for (char,idx) in self.latin_char2idx.items()}
                  shared.latin_char2idx = self.latin_char2idx
                  shared.latin_idx2char = self.latin_idx2char

            else:
                  self.native_char2idx = shared.native_char2idx 
                  self.native_idx2char = shared.native_idx2char 
                  self.latin_char2idx = shared.latin_char2idx 
                  self.latin_idx2char = shared.latin_idx2char 

            
            """ Torch tensors to make the data compatible for training """
            self.native_tensors = [torch.tensor(self.encode(native, self.native_char2idx)) for native in self.native_words]
            self.latin_tensors = [torch.tensor(self.encode(latin, self.latin_char2idx)) for latin in self.latin_words]

            self.data = [(native, latin) for native, latin in zip(self.native_words, self.latin_words)]
            

      def get_word_list(self, path):
            """ Function for retrieveing the vocabulary from the list of native and latin words """
            native_words, latin_words = [], []
            with open(path, encoding = 'utf-8') as f:
                  for i, row in enumerate(f):
                        line = row.strip().split()                        
                        if len(line) < 2:
                              continue
                        native, latin = line[0], line[1]
                        native_words.append(native)
                        latin_words.append(latin)
            return native_words, latin_words

      def char2idx(self, words):

            """ Function for generating a one-one mapping from characters to index """
            chars = sorted(set("".join(words)))
            char2idx = {'<pad>':0, '<unk>':1, '<sos>':3, '<eos>':4}
            char2idx.update({char:i+4 for i,char in enumerate(chars)})
            return char2idx

      def encode(self, word, vocab, max_len=10):

            """ Function for encoding the words into tokens to make it compatible for torch operations """
            encoded = [vocab.get(c, vocab['<unk>']) for c in word] + [vocab['<eos>']]
            encoded = encoded[:max_len] + [vocab['<pad>']]*(max_len - len(encoded))
            return encoded

      def decode(self, token, idx2char):

            """ Function for decoding the tokens into words to ensure fidelity of encoding """
            decoded = [idx2char.get(idx.item()) for idx in token]
            return ''.join(i for i in decoded if not i in [None, '<pad>', '<unk>'])

      def __len__(self):
            return len(self.data)
      
      def __getitem__(self, index):
            return self.native_tensors[index], self.latin_tensors[index]


                  
