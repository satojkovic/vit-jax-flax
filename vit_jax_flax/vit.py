import jax.numpy as jnp
import math
from flax import linen as nn
from typing import Optional


class Patches(nn.Module):
    patch_size: int
    embed_dim: int

    def setup(self):
        self.conv = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID'
        )

    def __call__(self, images):
        patches = self.conv(images)
        b, h, w, c = patches.shape
        patches = jnp.reshape(patches, (b, h*w, c))
        return patches


class PatchEncoder(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 3
        n, seq_len, _ = x.shape
        # Hidden dim
        x = nn.Dense(self.hidden_dim)(x)
        # Add cls token
        cls = self.param(
            'cls_token',
            nn.initializers.zeros,
            (1, 1, self.hidden_dim)
        )
        cls = jnp.tile(cls, (n, 1, 1))
        x = jnp.concatenate([cls, x], axis=1)
        # Add position embedding
        pos_embed = self.param(
            'position_embedding',
            nn.initializers.normal(stddev=0.02),  # From BERT
            (1, seq_len + 1, self.hidden_dim)
        )
        return x + pos_embed


class MultiHeadSelfAttention(nn.Module):
    hidden_dim: int
    n_heads: int
    drop_p: float

    def setup(self):
        self.q_net = nn.Dense(self.hidden_dim)
        self.k_net = nn.Dense(self.hidden_dim)
        self.v_net = nn.Dense(self.hidden_dim)

        self.proj_net = nn.Dense(self.hidden_dim)

        self.att_drop = nn.Dropout(self.drop_p)
        self.proj_drop = nn.Dropout(self.drop_p)

    def __call__(self, x, train=True):
        B, T, C = x.shape  # batch_size, seq_length, hidden_dim
        N, D = self.n_heads, C // self.n_heads  # num_heads, head_dim
        q = self.q_net(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)  # (B, N, T, D)
        k = self.k_net(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        v = self.v_net(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)

        # weights (B, N, T, T)
        weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(D)
        normalized_weights = nn.softmax(weights, axis=-1)

        # attention (B, N, T, D)
        attention = jnp.matmul(normalized_weights, v)
        attention = self.att_drop(attention, deterministic=not train)

        # gather heads
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N*D)

        # project
        out = self.proj_drop(self.proj_net(attention), deterministic=not train)

        return out


class MLP(nn.Module):
    mlp_dim: int
    drop_p: float
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs, train=True):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.mlp_dim)(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.drop_p, deterministic=not train)(x)
        x = nn.Dense(features=actual_out_dim)(x)
        x = nn.Dropout(rate=self.drop_p, deterministic=not train)(x)
        return x


class Transformer(nn.Module):
    embed_dim: int
    hidden_dim: int
    n_heads: int
    drop_p: float
    mlp_dim: int

    def setup(self):
        self.mha = MultiHeadSelfAttention(self.hidden_dim, self.n_heads, self.drop_p)
        self.mlp = MLP(self.mlp_dim, self.drop_p)
        self.layer_norm = nn.LayerNorm(epsilon=1e-6)
        self.dropout = nn.Dropout(rate=self.drop_p)

    def __call__(self, inputs, train=True):
        # Attention Block
        x = self.layer_norm(inputs)
        x = self.mha(x, train)
        x = inputs + self.dropout(x, deterministic=not train)
        # MLP block
        y = self.layer_norm(x)
        y = self.mlp(y, train)

        return x + y


class ViT(nn.Module):
    patch_size: int
    embed_dim: int
    hidden_dim: int
    n_heads: int
    drop_p: float
    num_layers: int
    mlp_dim: int
    num_classes: int

    def setup(self):
        self.patch_extracter = Patches(self.patch_size, self.embed_dim)
        self.patch_encoder = PatchEncoder(self.hidden_dim)
        self.transformer = Transformer(self.embed_dim, self.hidden_dim, self.n_heads, self.drop_p, self.mlp_dim)
        self.mlp_head = MLP(self.mlp_dim, self.drop_p)
        self.cls_head = nn.Dense(features=self.num_classes)

    def __call__(self, x, train=True):
        x = self.patch_extracter(x)
        x = self.patch_encoder(x)
        for i in range(self.num_layers):
            x = self.transformer(x, train)
        # MLP head
        x = x[:, 0]  # [CLS] token
        x = self.mlp_head(x, train)
        x = self.cls_head(x)
        return x
