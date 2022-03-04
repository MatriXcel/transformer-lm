#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Vladislav Lialin and Namrata Shivagunde 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lm.modeling_attention import MultiHeadSelfAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, fcn_hidden, dropout=0.0, causal=False):
        super().__init__()
        # Task 2.1 (1 point)
        # Create layers needed for Transformer Layer
        # 1. Create self.self_attention using MultiHeadSelfAttention, pass causal to it
        # 2. Create self.att_layer_norm using LayerNorm
        # 3. Create self.fcn using nn.Sequential, nn.ReLU and nn.Linear
        # 4. Create self.fcn_layer_norm using LayerNorm
        # 5. Create self.dropout using nn.Dropout
        #
        # YOUR CODE STARTS HERE  (our implementation is about 5-8 lines)    
        self.self_attention = MultiHeadSelfAttention(hidden, hidden, num_heads, causal, dropout)
        self.att_layer_norm = torch.nn.LayerNorm(hidden)

        self.fcn = nn.Sequential(nn.Linear(hidden, fcn_hidden), nn.ReLU(), nn.Linear(fcn_hidden, hidden))
        self.fcn_layer_norm = torch.nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        # YOUR CODE ENDS HERE

    def forward(self, x):
        """Self-Attention -> residual -> LayerNorm -> FCN -> residual -> LayerNorm
        
        Args:
            x: FloatTensor[batch_size, seq_len, input_size]
        
        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """

        # Task 2.2 (2 points)
        # Implement Transformer encoder block
        # Remember that transformer encoder block is composed of:
        # 1. Self-Attention
        # 2. Residual connection
        # 3. LayerNorm
        # 4. Fully-Connected Layer
        # 5. Dropout
        # 6. Residual connection
        # 7. LayerNorm
        # Note : Please write shape of the tensor for each line of code
        # YOUR CODE STARTS HERE (our implementation is about 6 lines)
        self.self_attention = self.self_attention.to(x.device)
        self.att_layer_norm = self.att_layer_norm.to(x.device)
        self.fcn = self.fcn.to(x.device)

        h = self.self_attention(x)
        h = h + x

        h = self.att_layer_norm(h)

        h2 = self.fcn(h)
        h2 = h2 + h
        h2 = self.att_layer_norm(h2)


        # YOUR CODE ENDS HERE
        return h2


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden, num_heads, fcn_hidden, vocab_size, max_seq_len, dropout=0.1, causal=False):
        """A minimal implementation of Transformer Encoder
        
        Args:
            num_layer: number of layers for encoder and decoder (in total, model will have 2 * num_layers layers)
            hidden: embedding size and hidden size of attentions
            fcn_hidden: hidden size of fully-connected networks inside transformer layers
            vocab_size: size of vocabulary
            max_seq_len: maximum length of input sequence
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden = hidden
        self.num_heads = num_heads
        self.fcn_hidden = fcn_hidden
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout

        # Task 2.3 (1 point)
        # 1. Create embedding layer and positional embedding layer
        # Use nn.Embedding for that
        # 2. Create a linear layer logit_proj that will project contextualized representations
        # of size hidden to your vocabulary size.
        # 3. Create a dropout layer
        # YOUR CODE STARTS HERE (our implementation is about 4 lines)
        self.encoder_emb = nn.Embedding(vocab_size, hidden)
        self.positional_emb = nn.Embedding(vocab_size, hidden)

        self.logit_proj = nn.Linear(hidden, vocab_size)
        self.dropout = nn.Dropout(dropout)
        # YOUR CODE ENDS HERE

        # Task 2.4 (1 point)
        # Create a list of encoder Layers
        # Note that you need to wrap it with nn.ModuleList,
        # so that the parameters of the layers would be counted as the paramertes of the model
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # Read more about ModuleList here:
        # https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict
        # You can use for-loop of python list comprehension to create the list of layers
        #
        # YOUR CODE STARTS HERE (our implementation is 1-3 lines)
        self.encoder_layers = nn.ModuleList([ TransformerEncoderLayer(hidden, num_heads, fcn_hidden, dropout) ] * num_layers)
        # YOUR CODE ENDS HERE

    def _add_positions(self, sequence_tensor):
        """Adds positional embeddings to the input tensor.

        Args:
            sequence_tensor: FloatTensor[batch_size, seq_len, hidden]
        """
        seq_len = sequence_tensor.shape[1]

        # Task 2.5 (1 point)
        # Use torch.arange to create a tensor of consequent numbers 1,2,3... of size [seq_len,]
        # specify the device of this tensor to be the same as sequence_tensor.device
        # Embed these tensors using positional embedding layer
        # and add them to the input tensor
        # YOUR CODE STARTS HERE (our implementation is about 3 lines)
        seq_range = torch.arange(1, seq_len+1)
        seq_range = seq_range.to(sequence_tensor.device)
        self.positional_emb = self.positional_emb.to(sequence_tensor.device)
        position_emb = self.positional_emb(seq_range)
        position_emb = position_emb.to(sequence_tensor.device)

        # YOUR CODE ENDS HERE

        return sequence_tensor + position_emb

    def forward(self, input_ids=None):
        """
        Args:
            input_ids: LongTensor[batch_size, src_seq_len], optional, encoder_embeds could be used instead
            input_ids: LongTensor[batch_size, tgt_seq_len]
            encoder_embeds: FloatTensor[batch_size, src_seq_len, hidden], cached output of self.encoder_layers, it is used in .generate
            key_padding_mask: padding mask for source sequence
        """
        # Task 2.6 (2 points)
        # Implement Transformer Encoder
        # Remember that Transformer Encoder is composed of:
        # 1. Embedding
        # 2. Positional Embedding (use self._add_positions)
        # 3. Transformer Encoder Layers
        # NOTE: Please write shape of the tensor for each line of code
        # YOUR CODE STARTS HERE(our implementation is about 4 lines)
        self.encoder_emb = self.encoder_emb.to(input_ids.device)
        embeds = self.encoder_emb(input_ids)
        embed = embeds.to(input_ids.device)
        embeds_with_pos = self._add_positions(embeds)
        
        
        for i, l in enumerate(self.encoder_layers):
            output = l(embeds_with_pos)
            #.to(input_ids.device)

        # YOUR CODE ENDS HERE
        return output


class TransformerLM(nn.Module):
    def __init__(self, num_layers, hidden, num_heads, fcn_hidden, vocab_size, max_seq_len, dropout=0.1):
        """Transformer Language Model"""
        super().__init__()
        self.dropout_rate = dropout

        # Task 2.7 (1 point)
        # Create a Transformer Encoder, output layer for language modeling, and a dropout layer
        # Remember that when we use Transformer for language modeling, it should be **causal** or it will cheat.
        # Output layer should predict the logits for all words in the vocabulary (size of logits = vocab_size)
        # YOUR CODE STARTS HERE (our implementation is about 2 lines)
        self.encoder = TransformerEncoder(num_layers, hidden, num_heads, fcn_hidden, vocab_size, max_seq_len, dropout, causal=True)
        self.out_proj = nn.Linear(vocab_size, vocab_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        # YOUR CODE ENDS HERE
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: LongTensor[batch_size, src_seq_len], optional, encoder_embeds could be used instead
        """
        assert input_ids.dim() == 2, "Input should be of size [batch_size, seq_len]"
        # Task 2.8 (1 point)
        # Implement Transformer Language Model
        # Remember that Transformer Language Model is composed of:
        # 1. Transformer Encoder
        # 2. Dropout
        # 3. Output Layer to produce logits over the classes (our vocabulary in case of language modeling)
        # YOUR CODE STARTS HERE (our implementation is 2 lines)
        self.encoder.logit_proj.to(input_ids.device)
        
        encoder_embeds = self.encoder.logit_proj(self.encoder(input_ids))
        encoder_embeds = self.dropout(encoder_embeds)

        logits = self.out_proj(encoder_embeds)
        # YOUR CODE ENDS HERE
        return logits

    @torch.inference_mode()
    def generate(self, input_ids, max_length):
        """
        Args:
            encoder_inputs: LongTensor[batch_size, seq_len]
        Returns:
            LongTensor[batch_size, max_length] — indices of generated words
        """
        for _ in range(max_length):
            logits = self(input_ids)
            last_word_logit = logits[:, -1, :]

            _, predicted_tokens = last_word_logit.max(dim=-1)
            input_ids = torch.cat([input_ids, predicted_tokens.unsqueeze(1)], dim=1)

        return input_ids[:, 1:]


    def save_pretrained(self, save_path):
        """Save the model weights to a directory

        Args:
            save_path: directory to save the model
        """
        config = {
            "num_layers": self.encoder.num_layers,
            "hidden": self.encoder.hidden,
            "num_heads": self.encoder.num_heads,
            "fcn_hidden": self.encoder.fcn_hidden,
            "vocab_size": self.encoder.vocab_size,
            "max_seq_len": self.encoder.max_seq_len,
            "dropout": self.encoder.dropout_rate,
        }

        with open(os.path.join(save_path, "model_config.json"), "w") as f:
           json.dump(config, f)

        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_path, "model.pt"))
    
    @classmethod
    def from_pretrained(cls, save_path):
        """Load the model weights from a directory

        Args:
            save_path: directory to load the model
        """
        with open(os.path.join(save_path, "model_config.json"), "r") as f:
            config = json.load(f)
        
        model = cls(**config)
        state_dict = torch.load(os.path.join(save_path, "model.pt"))
        model.load_state_dict(state_dict)
        return model
