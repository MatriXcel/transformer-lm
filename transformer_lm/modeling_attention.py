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

from xml.sax.xmlreader import AttributesNSImpl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden):
        """Self-attention module which computes softmax(xQ @ xK^T) @ xV

        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
        """
        super().__init__()
        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.scale = hidden ** 0.5              

    def forward(self, x):
        """Softmax(xQ @ xK^T) @ xV

        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        # Task 1.1: Compute Self Attention (3 points)
        # 1. Compute key, query and value matrices from your input x using self.k, self.q and self.v
        # 2. Compute the scores using query and key matrices
        # 3. Compute probabilities using softmax and scale the scores using self.scale
        # 4. Compute the output using probabilities and value matrices
        #
        # Please write shape of each tensor for each line of code
        # for example:
        #       Suppose batch_size = 3 and seq_len = 5
        #       x = torch.zeros(3, 5g) # shape [batch_size, seq_len] 
        #       x = x.unqueeze(1)     # shape [batch_size, 1, seq_len]
        # 
        # NOTE: Remmenber that we work with batches of data [batch_size, seq_len, hidden],
        # not just single examples [seq_len, hidden] as we did in the lecture. This changes your shapes a bit.
        #
        # YOUR CODE STARTS HERE (~ can be implemented in 4 lines or 3 if you combine steps 2 and 3 into one operation)
        k = self.k(x) #[batch_size, seq_len, hidden_dim] 
        q = self.q(x) #[batch_size, seq_len, hidden_dim]
        v = self.v(x) #[batch_size, seg_len, hidden_dim]

        scores = q @ k.transpose(1, 2) #[batch_size, seq_len, seq_len]
        
        softmaxOp = torch.nn.Softmax(dim=2) 
        probs = softmaxOp(scores/self.scale) #[batch_size, seq_len, seq_len]

        attention = probs @ v #[batch_size, seq_len, hidden_dim]

        # YOUR CODE ENDS HERE

    


        return attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, hidden, num_heads, causal=False, dropout=0):
        """
        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
            num_heads: int, number of attention heads, should be a divisor of hidden
            causal: use causal masking (do not allow target to look to the future or current token of source)
        """
        if hidden % num_heads:
            raise ValueError(f"hidden should be divisible by num_heads, "
                             f"but got hidden={hidden} and num_heads={num_heads}")
        super().__init__()

        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.head_size = hidden // num_heads
        self.scale = self.head_size ** 0.5

        self.causal = causal  # causal masking

    def forward(self, x, return_attention=False):

        """Computes [Softmax(x Q_1 @ x K_1^T) @ x V_1 : ... : Softmax(x Q_heads @ x K_heads^T) @ x V_heads] @ U

        or in more details:
        [SelfAttention_1(x) : ... : SelfAttention_h(x)] @ U

        where SelfAttention(x) = Softmax(x Q @ x K^T) @ x V
        and [:] is a concatenation operation.

        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        bs, seq, _ = x.shape
 
        # Task 1.2  (1 point)
        # 1. Compute key, query and value matrices from your input x using self.k, self.q and self.v
        # 1. Split them into multiple heads for multihead attention.
        # This can be achieves as a sequence of transpose and reshape operations.
        # Hint: Your target shape is [batch * num_heads, seq, hidden / num_heads]
        # 
        # NOTE: notice that reshape and transpose operations are different,
        # for example, given a tensor of shape [M, N] .reshape (N, M) and .transpose (1, 0)
        # will return you **different** tensors even thought their shapes are the same.
        # https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/
        #
        # Please write how the shape of the tensors are changing after each operation
        # for example:
        #        suppose 'a' is a tensor of shape [batch_size, input_size] and we apply following operation on it
        #        a = a.unsqueeze(1).transpose() # shape [batch_size, input_size, 1] 
        #        # 'a' [batch_size, input_size] -> unsqueeze [batch_size, 1, input_size] -> transpose [batch_size, input_size, 1]
        #
        # This comment explains how each operation is changing the shape of the tensor 'a' before it finally gets assigned to 'a' again.
        # This is just an example and does not suggest if these operations are used for this task.
        # YOUR CODE STARTS HERE (Our implementation is in 3 lines, one for each for k, q and v)

        # YOUR CODE ENDS HERE

        # TASK 1.3 (1 point)
        # 1. Compute scores (query key product) and scale them
        # YOUR CODE STARTS HERE  (our implementation is in 1 line)

        k = self.k(x) #[batch_size, seq_len, hidden_dim] 
        q = self.q(x) #[batch_size, seq_len, hidden_dim] 
        v = self.v(x) #[batch_size, seq_len, hidden_dim] 

        
        batch_combined_k = k.transpose(1, 2).reshape((bs, self.num_heads, self.head_size, seq)).transpose(2, 3)\
        .reshape((bs * self.num_heads, seq, self.head_size)) #[batch_size * num_heads, seq_len, head_size]

        batch_combined_q =  q.transpose(1, 2).reshape((bs, self.num_heads, self.head_size, seq)).transpose(2, 3)\
        .reshape((bs * self.num_heads, seq, self.head_size)) #[batch_size * num_heads, seq_len, head_size]
        batch_combined_v =  v.transpose(1, 2).reshape((bs, self.num_heads, self.head_size, seq)).transpose(2, 3)\
            .reshape((bs * self.num_heads, seq, self.head_size)) #[batch_size * num_heads, seq_len, head_size]
        
        
        scores = batch_combined_q @ batch_combined_k.transpose(1,2) #[batch_size * num_heads, seq_len, seq_len]
        # YOUR CODE ENDS HERE

        if self.causal:
            # Task 1.4 (2 points)
            # Apply casual mask to the scores
            # 1. Create a casual_mask that does not allow queries to look to the future keys
            # Specify the device of the causal_mask tensor to be the same as the scores.device
            # 2. Apply this casual mask on scores, fill with '-inf' where the mask is present.
            # 
            # You will find the following functions useful:
            #  - torch.triu
            #  - .masked_fill_
            #
            casual_mask = (torch.triu(scores, diagonal=1) != 0)
            casual_mask.to(scores.device)

            scores = scores.masked_fill(casual_mask, -torch.inf)
            # YOUR CODE ENDS HERE

        # Task 1.5 (2 point)
        # Compute probability (probs) and attention (att)
        # 1. Compute probabilities using scores, name them `probs`
        # 2. Apply dropout to the computed probabilities (a common place to put dropout in)
        # 3. Compute attention using probabilities
        # 4. Apply a number of matrix transformation on attention to change its dimenion to [batch, seq, hidden] (Our implmentation has four operations)
        # 5. Mix the attentions using self.mix, name the output `att`
        # Please write shape of the tensor for each line of code
        # 
        # NOTE: correct shapes do not guarantee correctness of the code.
        # You should understand how the reshapes and transposes are changing the elements of the tensor.
        #
        # YOUR CODE STARTS HERE (can be implemented in 4 lines)

        softmaxOp = torch.nn.Softmax(dim=2) #[batch_size * num_heads, seq_len, seq_len]
        probs = softmaxOp(scores / self.scale) #[batch_size * num_heads, seq_len, seq_len]

        probs = self.dropout(probs) #[batch_size * num_heads, seq_len, seq_len]
        attention = probs @ batch_combined_v #[batch_size * num_heads, seq_len, head_size]

        attention = attention.reshape((bs, self.num_heads, seq, self.head_size)).transpose(1, 2)\
        .reshape((bs, seq, self.head_size * self.num_heads))  #[batch_size, seq_len, hidden_dim]

        att = self.mix(attention) #[batch_size, seq_len, hidden_dim]
        
        # YOUR CODE ENDS HERE

        if return_attention:
            return att, probs

        return att
