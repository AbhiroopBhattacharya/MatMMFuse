import torch
import torch.nn as nn

class AttentionCombiner(nn.Module):
        def __init__(self, dim):
            super(AttentionCombiner, self).__init__()
            self.query = nn.Linear(dim, dim)  # Query for attention
            self.key = nn.Linear(dim, dim)  # Key for attention
            self.value = nn.Linear(dim, dim)  # Value for attention
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, supervised_embedding, transformer_embedding):
            query = self.query(transformer_embedding)
            key = self.key(supervised_embedding)
            value = self.value(supervised_embedding)
            # Compute attention scores
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.size(-1) ** 0.5)
            attention_weights = self.softmax(attention_scores)
            # Weighted sum of values
            combined = torch.matmul(attention_weights, value)
            return combined.squeeze(1)

class MultiHeadAttentionCombiner(nn.Module):
        def __init__(self, dim, num_heads=8):
            super(MultiHeadAttentionCombiner, self).__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads

            assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.fc_out = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, supervised_embedding, transformer_embedding):
            batch_size = supervised_embedding.size(0)

            # Linear transformations
            query = self.query(transformer_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)
            key = self.key(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention_weights = self.softmax(attention_scores)
            combined = torch.matmul(attention_weights, value)

            # Concatenate heads and apply final linear layer
            combined = combined.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
            combined = self.fc_out(combined)

            return combined.squeeze(1)

class AttentionCombinerWithNorm(nn.Module):
        def __init__(self, dim):
            super(AttentionCombinerWithNorm, self).__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, supervised_embedding, transformer_embedding):
            # Layer normalization
            supervised_embedding = self.norm1(supervised_embedding)
            transformer_embedding = self.norm1(transformer_embedding)

            # Linear transformations
            query = self.query(transformer_embedding)
            key = self.key(supervised_embedding)
            value = self.value(supervised_embedding)

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.size(-1) ** 0.5)
            attention_weights = self.softmax(attention_scores)
            combined = torch.matmul(attention_weights, value)

            # Layer normalization
            combined = self.norm2(combined)

            return combined.squeeze(1)

class AttentionCombinerWithResidual(nn.Module):
        def __init__(self, dim):
            super(AttentionCombinerWithResidual, self).__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)
            self.norm = nn.LayerNorm(dim)

        def forward(self, supervised_embedding, transformer_embedding):
            # Save the input for the residual connection
            residual = transformer_embedding

            # Linear transformations
            query = self.query(transformer_embedding)
            key = self.key(supervised_embedding)
            value = self.value(supervised_embedding)

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.size(-1) ** 0.5)
            attention_weights = self.softmax(attention_scores)
            combined = torch.matmul(attention_weights, value)

            # Add residual connection and apply layer normalization
            combined = self.norm(combined + residual)

            return combined.squeeze(1)

class AttentionCombinerWithDropout(nn.Module):
        def __init__(self, dim, dropout=0.2):
            super(AttentionCombinerWithDropout, self).__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, supervised_embedding, transformer_embedding):
            # Linear transformations
            query = self.query(transformer_embedding)
            key = self.key(supervised_embedding)
            value = self.value(supervised_embedding)

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.size(-1) ** 0.5)
            attention_weights = self.softmax(attention_scores)
            attention_weights = self.dropout(attention_weights)  # Apply dropout
            combined = torch.matmul(attention_weights, value)

            return combined.squeeze(1)

class ImprovedAttentionCombiner(nn.Module):
        def __init__(self, dim, num_heads=8, dropout=0.2):
            super(ImprovedAttentionCombiner, self).__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads

            assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.fc_out = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, supervised_embedding, transformer_embedding, node_mask=None):
            batch_size = supervised_embedding.size(0)

            # Layer normalization
            supervised_embedding = self.norm1(supervised_embedding)
            transformer_embedding = self.norm1(transformer_embedding)

            # Linear transformations
            query = self.query(transformer_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)
            key = self.key(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
                raise ValueError("NaN or Inf detected in attention_scores")
            if node_mask is not None:
                attention_scores = attention_scores.masked_fill(~node_mask[:, None, None, :],float("-inf"))
            attention_weights = self.softmax(attention_scores)
            attention_weights = self.dropout(attention_weights)  # Apply dropout
            combined = torch.matmul(attention_weights, value)

            # Concatenate heads and apply final linear layer
            combined = combined.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
            combined = self.fc_out(combined)

            # Add residual connection and apply layer normalization
            combined = self.norm2(combined + transformer_embedding)

            return combined.squeeze(1)
