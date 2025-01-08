
import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type
from .common import LayerNorm2d
import torch.nn.functional as F
class AutoPromptGenerator(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        batchsize: int = 8,
        task_number: int = 2,
        device: str = 'cuda'
    ) -> None:

        """
        Auto generate prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim

        self.mask_adapter = FourConv(in_channels=embed_dim,out_channels=embed_dim, mid_channels=embed_dim//4)
        self.task_output_attn_blocks = nn.ModuleList()
        self.image_output_attn_blocks = nn.ModuleList()
        self.task_number = task_number
        self.device = device

        self.task_output_common_mlp = nn.Linear(embed_dim, embed_dim)
        self.task_tokens = nn.Parameter(torch.randn(1,task_number, embed_dim)).to(device)  # 1 task_n dim


        for i in range(depth):
            task_output_attn_block = DoubleAttnBlock(
                dim = embed_dim,
                num_heads = num_heads,
                mlp_ratio= mlp_ratio)
            image_output_attn_block = DoubleAttnBlock(
                dim = embed_dim,
                num_heads = num_heads,
                mlp_ratio= mlp_ratio)
            self.task_output_attn_blocks.append(task_output_attn_block)
            self.image_output_attn_blocks.append(image_output_attn_block)



    def forward(self,image_embeddings,output_tokens):



        batchsize = image_embeddings.size(0)
        output_tokens = output_tokens.unsqueeze(0).expand(batchsize, -1, -1).unsqueeze(1).to(self.device) # b 1 5 256

        ori_output_tokens = output_tokens
        ori_task_tokens = task_tokens = self.task_tokens.expand(batchsize,-1,-1,-1) # 原始task_tokens 维度 1 task 256

        for blk in self.task_output_attn_blocks:
            task_tokens,output_tokens = blk(task_tokens,output_tokens)

        # 更新task_tokens 和 output_tokens
        task_tokens = self.task_output_common_mlp(task_tokens)+ori_task_tokens # b 1 task 256
        output_tokens = self.task_output_common_mlp(output_tokens)+ori_output_tokens # b 1 5 256

        image_embeddings = image_embeddings.permute(0,2,3,1)
        T = torch.concat([task_tokens,output_tokens],dim=-2)
        for blk in self.image_output_attn_blocks:
            image_embeddings,T = blk(image_embeddings,T)
        image_embeddings = image_embeddings.permute(0,3,1,2)
        Ps = T[:,:,:self.task_number,:].squeeze(1)

        Pd = self.mask_adapter(image_embeddings)

        return image_embeddings,Ps,Pd

class DoubleAttnBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.q_kv_cross_attn = qkvAttention(dim=dim, num_heads=num_heads)  # with skip connection
        self.q_kv_mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.kv_q_cross_attn = qkvAttention(dim=dim, num_heads=num_heads)  # with skip connection
        self.kv_q_mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.fusion_q_kv_cross_attn = qkvAttention(dim=dim, num_heads=num_heads)  # with skip connection
        self.fusion_q_kv_mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.fusion_kv_q_cross_attn = qkvAttention(dim=dim, num_heads=num_heads)  # with skip connection
        self.fusion_kv_q_mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        q_1,kv_1 = x1,x2
        q_2,kv_2 = x2,x1

        x_1 = self.q_kv_cross_attn(q_1,kv_1,kv_1)
        x_1 = self.q_kv_mlp(x_1)

        x_2 = self.kv_q_cross_attn(q_2,kv_2,kv_2)
        x_2 = self.kv_q_mlp(x_2)

        fusion_x1 = self.fusion_q_kv_cross_attn(x_1,x_2,x_2)
        fusion_x1 = self.fusion_q_kv_mlp(fusion_x1)

        fusion_x2 = self.fusion_kv_q_cross_attn(x_2,x_1,x_1)
        fusion_x2 = self.fusion_kv_q_mlp(fusion_x2)

        return fusion_x1, fusion_x2
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x
        #return self.lin2(self.act(self.lin1(x)))
class qkvAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = q.shape
        _,H2,W2,_ = k.shape

        q = self.q(q).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads,
                                                                                        H * W, -1)
        k = self.k(k).reshape(B, H2 * W2, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads,
                                                                                        H2 * W2, -1)
        v = self.v(v).reshape(B, H2 * W2, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads,
                                                                                        H2 * W2, -1)
        # print(q.shape)
        # print(k.shape)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


class FourConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.fourth_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.fourth_conv(x)