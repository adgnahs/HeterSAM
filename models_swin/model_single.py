from models_swin.base import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, full_features, out):
        super(Decoder, self).__init__()
        # self.up1 = UpBlockSkip(full_features[4] + full_features[3], full_features[3],
        #                        func='relu', drop=0).cuda()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0).cuda()
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0).cuda()
        self.up3 = UpBlockSkip(full_features[1] + full_features[0], full_features[0],
                               func='relu', drop=0).cuda()
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.final = CNNBlock(full_features[0], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        #第14层和第9层传给upblockskip的forward
        #hardnet中特定层的特征图进行upblockskip
        z = self.up2(z, x[1])
        z = self.up3(z, x[0])
        # z = self.up4(z, x[0])
        z = self.Upsample(z)
        out = F.tanh(self.final(z))
        #连续三个拼接卷积dropout激活后再上采样最后激活
        return out


class SwinUNetDecoder(nn.Module):
    def __init__(self, embed_dims, depths, num_heads, window_size=7):
        """
        embed_dims: 通道数列表，如 [128, 256, 512, 1024]
        depths: 每层 Swin Block 数，如 [2, 2, 2, 2]
        num_heads: 每层 head 数，如 [4, 8, 16, 32]
        """
        super().__init__()
        self.num_layers = len(embed_dims)

        self.up_blocks = nn.ModuleList()
        self.concat_linear = nn.ModuleList()
        self.swin_blocks = nn.ModuleList()

        for i in range(self.num_layers - 1, 0, -1):  # 从 bottleneck 向上解码
            in_dim = embed_dims[i]
            skip_dim = embed_dims[i - 1]

            # PatchExpanding 将空间尺寸放大，通道变成一半（因为 concat 后变回）
            self.up_blocks.append(PatchExpanding(in_dim))

            # 将 concat 后的维度映射回 skip_dim
            self.concat_linear.append(nn.Linear(in_dim // 2 + skip_dim, skip_dim))

            # 构建 Swin Transformer Block（和 encoder 对称）
            self.swin_blocks.append(
                BasicLayer(
                    dim=skip_dim,
                    depth=depths[i - 1],
                    num_heads=num_heads[i - 1],
                    window_size=window_size,
                    downsample=None
                )
            )

    def forward(self, features):
        """
        features: list of stage1–4, from encoder
        每个 feature 是 (B, H, W, C)
        """
        x = features[-1]  # start from stage4 bottleneck
        for i in range(self.num_layers - 1):
            skip = features[-(i + 2)]  # stage3, stage2, stage1
            # 上采样
            x = self.up_blocks[i](x)  # 空间 ↑2，通道 ÷2
            # 拼接 skip connection
            x = torch.cat([x, skip], dim=-1)  # 在通道维度上 concat
            # Linear 映射回 skip_dim
            x = self.concat_linear[i](x)
            # Swin Transformer Block
            x = self.swin_blocks[i](x)

        return x  # 输出是最后一层的 token (B, H, W, C)


class Unet(nn.Module):
    def __init__(self, order, depth_wise, args):
        super(Unet, self).__init__()
        self.backbone = HarDNet(depth_wise=depth_wise, arch=order, args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = Decoder(d, out=1)
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img, size=None):
        z = self.backbone(img)
        M = self.decoder(z)
        return M


# class SmallDecoder(nn.Module):
#     def __init__(self, full_features, out):
#         super(SmallDecoder, self).__init__()
#         self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
#                                func='relu', drop=0).cuda()
#         self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
#                                func='relu', drop=0).cuda()
#         self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)
#
#     def forward(self, x):
#         z = self.up1(x[3], x[2])
#         z = self.up2(z, x[1])
#         out = F.tanh(self.final(z))
#         return out


# class SmallDecoder(nn.Module):
#     def __init__(self, full_features, out):
#         super(SmallDecoder, self).__init__()
#         self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
#                                func='relu', drop=0)
#         self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
#                                func='relu', drop=0)
#         self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)
#
#     def forward(self, x):
#         z = self.up1(x[3], x[2])
#         z = self.up2(z, x[1])
#         out = F.tanh(self.final(z))
#         # out = self.final(z)
#         return out

from timm.models.swin_transformer import SwinTransformer
import torch.nn as nn
from einops import rearrange


class MultiScaleSwin(SwinTransformer):
    def __init__(self, img_size=512, patch_size=4, in_chans=3,
                 embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
                 window_size=8, ** kwargs):

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            **  kwargs
        )

        # 重写初始投影
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size)
        )

        # 精确配置下采样层
        self.downsample_layers = nn.ModuleList([
            self._build_downsample(128, 256),
            self._build_downsample(256, 512),
            self._build_downsample(512, 1024)
        ])

        # 将父类的下采样替换为恒等映射
        for layer in self.layers:
            layer.downsample = nn.Identity()  # 避免NoneType错误

    def _build_downsample(self, in_dim, out_dim):
        """确保输入/输出维度精确匹配"""
        return nn.Sequential(
            nn.LayerNorm(4 * in_dim),
            nn.Linear(4 * in_dim, out_dim, bias=False)
        )

    def forward_features(self, x):
        features = []

        # 初始投影
        x = self.patch_embed(x)  # [B,3,512,512] → [B,128,128,128]
        # x = rearrange(x, 'b c h w -> b (h w) c')
        # print("========================0")
        # print(x.shape)
        B, C, H, W = x.shape
        # print(B,C,H,W)
        # H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).contiguous()

        # 处理各阶段
        for stage_idx in range(4):
            # print("这是第几轮：")
            # print(stage_idx)
            # 转换为空间格式
            B, H, W, C = x.shape

            # 通过Swin层（父类下采样已变为Identity）
            x = self.layers[stage_idx](x)

            # 转换为序列格式
            # x = rearrange(x, 'b h w c -> b (h w) c')
            features.append(x)  # 保存stage1-4
            # print("========================1")
            # print(x.shape)
            # 应用自定义下采样（前三个阶段）
            if stage_idx < 3:
                # 合并空间维度

                merged_dim = 4 * C
                # print(C)
                # print("----------")
                assert merged_dim == self.downsample_layers[stage_idx][0].normalized_shape[0], \
                    f"Stage {stage_idx} 通道不匹配: {merged_dim} vs {self.downsample_layers[stage_idx][0].normalized_shape}"
                # print("========================2")
                # print(x.shape)
                # 合并操作：2x2 -> 4C
                x = x.view(B, H // 2, 2, W // 2, 2, C)
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
                x = x.view(B, (H // 2) , (W // 2), merged_dim)
                # print("========================3")
                # print(x.shape)
                # 应用线性投影
                x = self.downsample_layers[stage_idx](x)


        return features

class ModelEmb(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = MultiScaleSwin(
            img_size=512,
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],  # 保持4个阶段
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.3,
            num_classes=0
        )
        # 更新特征配置
        self.full_features = [128, 256, 512, 1024]  # 对应stage1-4
        self.features = 1024
        self.decoder = SmallDecoder(
            full_features=self.full_features,
            out=256
        )

    def forward(self, img, size=None):
        features = self.backbone.forward_features(img)
        # print(img.shape)
        # 验证特征层级
        # print(f"特征层级数: {len(features)} → 预期4")
        # 选择需要的层级（示例取后4层）

        z = (features[1], features[2], features[3])

        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(
            dense_embeddings,
            (64, 64),
            mode='bilinear',
            align_corners=True
        )
        return dense_embeddings

class SmallDecoder(nn.Module):
    def __init__(self, full_features, out):
        super().__init__()
        # 假设使用stage2-4（索引1-3）
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], 512)
        self.up2 = UpBlockSkip(512 + full_features[1], 256)
        self.final = CNNBlock(256, out, 3)

    def forward(self, x):
        stage2, stage3, stage4 = x[0], x[1], x[2]  # 跳过初始层
        stage2 = stage2.permute(0, 3, 1, 2).contiguous()
        stage3 = stage3.permute(0, 3, 1, 2).contiguous()
        stage4 = stage4.permute(0, 3, 1, 2).contiguous()
        z = self.up1(stage4, stage3)
        z = self.up2(z, stage2)
        return F.tanh(self.final(z))

class SparseDecoder(nn.Module):
    def __init__(self, full_features, out, nP):
        super(SparseDecoder, self).__init__()
        self.final = CNNBlock(full_features[-1], 256, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.final(x[-1])
        out = z.reshape(8, 256, -1).permute(0, 2, 1)
        #通过 .reshape(8, 256, -1)，这一步将 z 的形状转换为 (8, 256, N)，其中 8 是批量大小，256 是输出通道数，N 是展开后的空间维度，-1 表示根据其他维度自动推算该维度的大小。
        #permute 函数用于重新排列张量的维度，在这里交换了第 2 和第 3 个维度
        return out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        nP = int(args['nP']) + 1
        half = 0.5*nP**2
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = Decoder(d, out=4)
        for param in self.backbone.parameters():
            param.requires_grad = True
        x = torch.arange(nP, nP**2, nP).long()
        y = torch.arange(nP, nP**2, nP).long()
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        P = torch.cat((grid_x.unsqueeze(dim=0), grid_y.unsqueeze(dim=0)), dim=0)
        P = P.view(2, -1).permute(1, 0).cuda()
        self.P = (P - half) / half
        pos_labels = torch.ones(P.shape[-2])
        neg_labels = torch.zeros(P.shape[-2])
        self.labels = torch.cat((pos_labels, neg_labels)).cuda().unsqueeze(dim=0)

    def forward(self, img, size=None):
        if size is None:
            half = img.shape[-1] / 2
        else:
            half = size / 2
        P = self.P.unsqueeze(dim=0).repeat(img.shape[0], 1, 1).unsqueeze(dim=1)
        z = self.backbone(img)
        J = self.decoder(z)
        dPx_neg = F.grid_sample(J[:, 0:1], P).transpose(3, 2)
        dPx_pos = F.grid_sample(J[:, 2:3], P).transpose(3, 2)
        dPy_neg = F.grid_sample(J[:, 1:2], P).transpose(3, 2)
        dPy_pos = F.grid_sample(J[:, 3:4], P).transpose(3, 2)
        dP_pos = torch.cat((dPx_pos, dPy_pos), -1)
        dP_neg = torch.cat((dPx_neg, dPy_neg), -1)
        P_pos = dP_pos + P
        P_neg = dP_neg + P
        P_pos = P_pos.clamp(min=-1, max=1)
        P_neg = P_neg.clamp(min=-1, max=1)
        points_norm = torch.cat((P_pos, P_neg), dim=2)
        points = (points_norm * half) + half
        return points, self.labels, J, points_norm


# class ModelEmb(nn.Module):
#     def __init__(self, args):
#         super(ModelEmb, self).__init__()
#         self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
#
#         #full_features：获取 HarDNet 提取的 完整特征通道数；features：获取 HarDNet 的 压缩特征数
#         d, f = self.backbone.full_features, self.backbone.features
#
#         self.decoder = SmallDecoder(d, out=256)
#
#         #允许backbone训练
#         for param in self.backbone.parameters():
#             param.requires_grad = True
#
#     def forward(self, img, size=None):
#         z = self.backbone(img)
#         dense_embeddings = self.decoder(z)
#         dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=True)
#         return dense_embeddings



class ModelSparseEmb(nn.Module):
    def __init__(self, args):
        super(ModelSparseEmb, self).__init__()
        nP = int(args['nP'])
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SparseDecoder(d, out=1, nP=nP)
        for param in self.backbone.parameters():
            param.requires_grad = True
        # pos_labels = torch.ones(int(args['nP']))
        # neg_labels = torch.zeros(int(args['nP']))
        # self.labels = torch.cat((pos_labels, neg_labels)).cuda().unsqueeze(dim=0)

    def forward(self, img, size=None):
        z = self.backbone(img)
        sparse_embeddings = self.decoder(z)
        return sparse_embeddings


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskEncoder(nn.Module):
    def __init__(self):
        super(MaskEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.norm1 = LayerNorm2d(4)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(16)
        self.conv3 = nn.Conv2d(16, 256, kernel_size=1)

    def forward(self, mask):
        z = self.conv1(mask)
        z = self.norm1(z)
        z = self.gelu(z)
        z = self.conv2(z)
        z = self.norm2(z)
        z = self.gelu(z)
        z = self.conv3(z)
        return z


class ModelH(nn.Module):
    def __init__(self):
        super(ModelH, self).__init__()
        self.conv1 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1)
        self.norm1 = LayerNorm2d(64)
        self.gelu = nn.GELU()
        self.conv2 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, mask):
        z = self.conv1(mask, output_size=(128, 128))
        z = self.norm1(z)
        z = self.gelu(z)
        z = self.conv2(z, output_size=(256, 256))
        z = self.norm2(z)
        z = self.gelu(z)
        z = self.conv3(z)
        return z


if __name__ == "__main__":
    import argparse
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-nP', '--nP', default=10, help='image size', required=False)
    args = vars(parser.parse_args())

    # sam_args = {
    #     'sam_checkpoint': "/home/tal/MedicalSam/cp/sam_vit_h_4b8939.pth",
    #     'model_type': "vit_h",
    #     'generator_args': {
    #         'points_per_side': 8,
    #         'pred_iou_thresh': 0.95,
    #         'stability_score_thresh': 0.7,
    #         'crop_n_layers': 0,
    #         'crop_n_points_downscale_factor': 2,
    #         'min_mask_region_area': 0,
    #         'point_grids': None,
    #         'box_nms_thresh': 0.7,
    #     },
    #     'gpu_id': 0,
    # }

    model = ModelH().cuda()
    # x = torch.randn((3, 3, 256, 256)).cuda()
    # P = model(x)
    # sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    # sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    # pretrain = sam.prompt_encoder.mask_downscaling

    # model = MaskEncoder().cuda()
    # model.conv1.load_state_dict(pretrain[0].state_dict())
    # model.norm1.load_state_dict(pretrain[1].state_dict())
    # model.conv2.load_state_dict(pretrain[3].state_dict())
    # model.norm2.load_state_dict(pretrain[4].state_dict())
    # model.conv3.load_state_dict(pretrain[6].state_dict())
    x = torch.randn((4, 256, 64, 64)).cuda()
    z = model(x)
    print(z.shape)




