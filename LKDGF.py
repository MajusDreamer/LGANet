import torch.nn as nn
import torch
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,dim=2)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=proj_drop)
        self.stoken_size = stoken_size
        self.blockUp = nn.Sequential(
            nn.ConvTranspose2d(
                dim, dim,
                kernel_size=10,
                stride=8,
                padding=1,
                output_padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim)
        )

        if self.stoken_size[0] == 4:
            self.blockUp =  nn.Sequential(
                nn.ConvTranspose2d(
                    dim, dim,
                    kernel_size=8,
                    stride=4,
                    padding=2,
                    output_padding=0
                ),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
        )
        elif self.stoken_size[0] == 2:
            self.blockUp = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )

    def forward(self, x):
        _, _, H, W = x.shape
        h, w = self.stoken_size

        hh, ww = H // h, W // w

        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))

        stoken_features = self.stoken_refine(stoken_features)

        x = self.blockUp(stoken_features)

        return x
class LKDF(nn.Module):
    """Parametric Large-Kernel Depthwise Separable Convolution Module"""

    def __init__(self, in_channels, out_channels_list=None, num_layers=4):
        """
        Args:
            in_channels: Number of input channels
            out_channels_list: List of output channels for each layer
            num_layers: Number of layers
        """
        super().__init__()

        if out_channels_list is None:
            out_channels_list = [64, 128, 256, 512]

        assert len(out_channels_list) == num_layers, "The length of the channel list must equal the number of layers."

        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        current_channels = in_channels
        for i, out_channels in enumerate(out_channels_list):

            depconv = nn.Sequential(
                nn.Conv2d(
                    current_channels, current_channels*2, kernel_size=11,
                    stride=2, padding=5, groups=current_channels, bias=False
                ),
                nn.BatchNorm2d(current_channels*2),
                nn.ReLU(),
                nn.Conv2d(current_channels*2, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

            if current_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, kernel_size=1,
                              stride=2, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, kernel_size=1,
                              stride=2, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

            self.layers.append(depconv)
            self.downsample_layers.append(downsample)
            current_channels = out_channels

        self.grid1 = StokenAttention(64, stoken_size=[8, 8])
        self.grid2 = StokenAttention(128, stoken_size=[8, 8])

    def forward(self, x):
        outputs = []
        current_x = x

        for i, (depconv, downsample) in enumerate(zip(self.layers, self.downsample_layers)):

            _, C, _, _ = current_x.shape
            shortcut = downsample(current_x)

            if C == 64:
                current_x = self.grid1(current_x)
            elif C == 128:
                current_x = self.grid2(current_x)
            main_path = depconv(current_x)

            if shortcut.shape != main_path.shape:
                main_path = F.adaptive_avg_pool2d(main_path, shortcut.shape[2:])

            current_x = main_path + shortcut
            outputs.append(current_x)

        return outputs

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )
class LKD_FusionFeature(nn.Module):

    def __init__(self, num_features=4):
        super().__init__()
        self.eps = 1e-8
        self.num_features = num_features

        channels_list = [64, 128, 256, 512]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(channels_list[i],2, dtype=torch.float32), requires_grad=True)
            for i in range(num_features)
        ])

        self.ds_block = LKDF(64)

        self.post_conv1 = ConvBNReLU(64, 64, kernel_size=3)
        self.post_conv2 = ConvBNReLU(128, 128, kernel_size=3)
        self.post_conv3 = ConvBNReLU(256, 256, kernel_size=3)
        self.post_conv4 = ConvBNReLU(512, 512, kernel_size=3)


    def forward(self, x, *res_features):
        _, C, _, _ = x.shape

        ds_features = self.ds_block(x)

        fused_features = []

        for i, (res_feat, ds_feat) in enumerate(zip(res_features, ds_features)):
            # 确保特征图尺寸匹配
            if res_feat.shape[2:] != ds_feat.shape[2:]:
                # print("输出特征图不匹配")
                ds_feat = F.adaptive_avg_pool2d(ds_feat, res_feat.shape[2:])

            # 使用sigmoid将权重限制在0到1之间，然后乘以2，得到0到2之间的权重
            weights = torch.sigmoid(self.weights[i]) * 2
            weight_a = weights[:, 0].view(1, -1, 1, 1)
            weight_b = weights[:, 1].view(1, -1, 1, 1)

            fused_feat = weight_a * res_feat +  weight_b * ds_feat
            _, CC, _, _ = fused_feat.shape
            if CC == 64:
                # print(C,e)
                fused_feat = self.post_conv1(fused_feat)
            elif CC == 128:
                fused_feat = self.post_conv2(fused_feat)
            elif CC == 256:
                fused_feat = self.post_conv3(fused_feat)
            elif CC == 512:
                fused_feat = self.post_conv4(fused_feat)

            fused_features.append(fused_feat)

        return fused_features