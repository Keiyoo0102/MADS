import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from .segformer import Mlp, DWConv, Attention, Block, OverlapPatchEmbed

class TextVisualAlignmentModule(nn.Module):
    def __init__(self, vis_dim, text_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.vis_dim = vis_dim
        self.num_heads = num_heads
        self.scale = (vis_dim // num_heads) ** -0.5
        self.text_proj_k = nn.Linear(text_dim, vis_dim)
        self.text_proj_v = nn.Linear(text_dim, vis_dim)
        self.vis_proj_q = nn.Linear(vis_dim, vis_dim)
        self.out_proj = nn.Linear(vis_dim, vis_dim)
        self.norm = nn.LayerNorm(vis_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_feats, text_embeds):
        B, N, C = visual_feats.shape
        # Handle text embedding input (B, N_cls, D_text) or (N_cls, D_text)
        if text_embeds.dim() == 3:
            text_embeds_proto = text_embeds[0]
        else:
            text_embeds_proto = text_embeds

        N_cls, D_text = text_embeds_proto.shape
        q = self.vis_proj_q(visual_feats)
        text_expanded = text_embeds_proto.unsqueeze(0).expand(B, -1, -1)
        k = self.text_proj_k(text_expanded)
        v = self.text_proj_v(text_expanded)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N_cls, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N_cls, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = out + visual_feats
        out = self.norm(out)
        return out

class SegFormerHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=256, dropout_ratio=0.1):
        super().__init__()
        class MLP_Conv(nn.Module):
            def __init__(self, in_feat, out_feat):
                super().__init__()
                self.linear = nn.Conv2d(in_feat, out_feat, 1)
            def forward(self, x): return self.linear(x)

        self.linear_layers = nn.ModuleList([MLP_Conv(in_ch, embed_dim) for in_ch in in_channels])
        self.linear_fuse = nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, inputs):
        H, W = inputs[0].shape[2:]
        outs = []
        for i, x in enumerate(inputs):
            x = self.linear_layers[i](x)
            if x.shape[2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(x)
        out = torch.cat(outs, dim=1)
        out = self.linear_fuse(out)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear_pred(out)
        return out

class MixVisionTransformerFusion(nn.Module):
    def __init__(self, img_size=512, in_chans=64, num_classes=10, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], decoder_dim=256,
                 text_dim=512, fusion_stage=3):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.fusion_stage = fusion_stage
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size // (2 ** (i + 2)) if i > 0 else img_size,
                patch_size=7 if i == 0 else 3, stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1], embed_dim=embed_dims[i])
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i]) for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            self.patch_embeds.append(patch_embed)
            self.blocks.append(block)
            self.norms.append(norm)

        fusion_dim = self.embed_dims[fusion_stage]
        self.alignment_module = TextVisualAlignmentModule(vis_dim=fusion_dim, text_dim=text_dim)
        self.decode_head = SegFormerHead(in_channels=embed_dims, num_classes=num_classes, embed_dim=decoder_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, text_embeds):
        B = x.shape[0]
        outs = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]: x = blk(x, H, W)
            x = self.norms[i](x)
            if i == self.fusion_stage:
                x = self.alignment_module(x, text_embeds)
            C = x.shape[-1]
            x_spatial = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            outs.append(x_spatial)
            x = x_spatial
        out = self.decode_head(outs)
        out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
        return out

def get_fusion_model(num_classes=10):
    return MixVisionTransformerFusion(
        in_chans=64, num_classes=num_classes,
        embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        decoder_dim=256, text_dim=512, fusion_stage=3
    )