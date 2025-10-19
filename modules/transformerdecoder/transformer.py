import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class Attention(nn.Module):

    def __init__(self, dim, n_head, head_dim, dropout=0.):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.scale = head_dim ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1) # b h n n
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LocalAttention(nn.Module):
    def __init__(self, dim, window_size=2, n_head=4, head_dim=32):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads=n_head)

    def forward(self, x):
        B, N, D = x.shape  # Get batch size, sequence length (tokens), and feature dim
        W = self.window_size
        G = N // W  # Compute number of groups explicitly

        # Ensure N is divisible by W (should be true for window_size=2)
        assert N % W == 0, f"window_size={W} does not divide N={N} evenly!"

        # Apply Local Attention
        x = einops.rearrange(x, 'b (g w) d -> (b g) w d', g=G, w=W)  # Reshape for local attention
        x, _ = self.attn(x, x, x)
        x = einops.rearrange(x, '(b g) w d -> b (g w) d', g=G, w=W)  # Restore shape

        return x


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        Support additional arguments for wrapped modules.
        """
        return self.fn(self.norm(x), *args, **kwargs)


class TransformerEncoder(nn.Module):

    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ff(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Self-Attention for target sequence
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),

                # Cross-Attention between target and memory from encoder
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),

                # FeedForward layer
                PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
            ]))

    def forward(self, x, memory):
        """
        Args:
            x: Target sequence of shape (B, T, D)
            memory: Encoder outputs of shape (B, S, D)
        Returns:
            Decoded output of shape (B, T, D)
        """
        for norm_self_attn, norm_cross_attn, norm_ff in self.layers:
            # Self-attention over the target sequence
            x = x + norm_self_attn(x)

            # Cross-attention between target and encoder memory
            x = x + norm_cross_attn(x, to=memory)

            # Feedforward network
            x = x + norm_ff(x)
        
        return x



class Transformer(nn.Module):
    def __init__(self, dim, encoder_depth, decoder_depth, n_head, head_dim, ff_dim, dropout=0.):
        super().__init__()

        # Initialize the encoder and decoder
        self.encoder = TransformerEncoder(dim, encoder_depth, n_head, head_dim, ff_dim, dropout)
        self.decoder = TransformerDecoder(dim, decoder_depth, n_head, head_dim, ff_dim, dropout)

    def forward(self, src, tgt):
        """
        Args:
            src: Source sequence of shape (B, S, D) - Encoder input
            tgt: Target sequence of shape (B, T, D) - Decoder input
        Returns:
            Transformer output of shape (B, T, D)
        """
        # Encoder forward pass
        memory = self.encoder(src)

        # Decoder forward pass
        output = self.decoder(tgt, memory)

        return output
    
    def get_last_layer(self):
        """
        Returns the last layer weights of the decoder
        """
        return self.decoder.layers[-1][-1].fn.net[-2].weight