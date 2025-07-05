import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class InputLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        image_size: tuple,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = (192,256)

        # number of patches
        self.nb_patch = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        # split into patches
        self.patch_embed_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # cls token
        # self.cls_token = nn.parameter.Parameter(data=torch.randn(1, 1, self.embed_dim))
        # positional embedding
        self.positional_embedding = nn.parameter.Parameter(
            data=torch.randn(1, self.nb_patch, self.embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        # (Batch, Channel, Height, Width) -> (B, D, H/P, W/P)
        out = self.patch_embed_layer(x)
        # (B, D, H/P, W/P) -> (B, D, Np)
        # flatten from H/P(2) to W/P(3)
        out = torch.flatten(out, start_dim=2, end_dim=3)
        # (B, D, Np) -> (B, Np, D)
        out = out.transpose(1, 2)
        # # concat class token
        # # cat (B, 1, D), (B, Np, D) -> (B, Np + 1, D)
        # out = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), out], dim=1)
        # add positional embedding
        out += self.positional_embedding
        out = out.permute(0,2,1)
        out = out.view(B, self.embed_dim, H//self.patch_size, W//self.patch_size)
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, nb_head: int, dropout: float) -> None:
        super().__init__()
        self.nb_head = nb_head
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // nb_head

        self.w_q = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim, bias=False
        )
        self.w_k = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim, bias=False
        )
        self.w_v = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim, bias=False
        )
        self.w_o = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, nb_patch, _ = x.size()
        # (B, N, D) -> (B, nb_head, N, D//nb_head)
        # query
        q = self.w_q(x)
        q = q.view(batch_size, self.nb_head, nb_patch, self.head_dim)
        # key
        k = self.w_k(x)
        k = k.view(batch_size, self.nb_head, nb_patch, self.head_dim)
        # value
        v = self.w_v(x)
        v = v.view(batch_size, self.nb_head, nb_patch, self.head_dim)

        # inner product
        # (B, nb_head, N, D//nb_head) × (B, nb_head, D//nb_head, N) -> (B, nb_head, N, N)
        dots = (q @ k.transpose(2, 3)) / self.head_dim**0.5
        # softmax by columns
        # dim=3 eq dim=-1. dim=-1 applies softmax to the last dimension
        attn = F.softmax(dots, dim=3)
        # weighted
        # (B, nb_head, N, N) × (B, nb_head, N, D//nb_head) -> (B, nb_head, N, D//nb_head)
        out = attn @ v
        # (B, nb_head, N, D//nb_head) -> (B, N, nb_head, D//nb_head) -> (B, N, D)
        out = out.transpose(1, 2).reshape(batch_size, nb_patch, self.embed_dim)
        out = self.w_o(out)
        return out

class Block(nn.Module):
    def __init__(
        self, embed_dim: int, nb_head: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.msa = MultiHeadSelfAttention(
            embed_dim=embed_dim, nb_head=nb_head, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        # Reshape the tensor to (B,H*W,C)
        x= x.reshape(B, -1, C)
        # add skip-connect
        out = self.msa(self.ln1(x)) + x
        # add skip-connect
        out = self.mlp(self.ln2(out)) + out

        out = out.reshape(B,H,W,C)
        out = out.permute(0,3,1,2)
        return out