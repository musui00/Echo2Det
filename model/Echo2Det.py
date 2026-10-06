import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        return self.conv_block(x)

class MTFE(nn.Module):
    """Multi-scale Temporal Feature Extractor (MTFE)."""
    def __init__(self, out_channel):
        super().__init__()
        self.out_c = out_channel
        branch_channels = out_channel // 3
        
        self.conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(1, branch_channels, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(branch_channels)
        )
        
        self.conv_5 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, branch_channels, kernel_size=(2, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(branch_channels)
        )
        
        self.conv_7 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, branch_channels, kernel_size=(2, 7)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(branch_channels)
        )
        
        self.channel_attention = DPCA(out_channel)

    def forward(self, x):
        y1 = self.conv_3(x)
        y2 = self.conv_5(x)
        y3 = self.conv_7(x)
        return self.channel_attention(torch.cat([y1, y2, y3], dim=1))

class DPCA(nn.Module):
    """Dual-Pool Channel Attention (DPCA)."""
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class TCA(nn.Module):
    """Temporal Context Aggregation (TCA)."""
    def __init__(self, num_heads, input_size, hidden_size):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.num_heads = num_heads
        self.attn_head_size = hidden_size // num_heads
        self.all_head_size = self.attn_head_size * num_heads
        
        self.temporal_qkv = nn.Linear(input_size, 3*self.all_head_size)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # === 时间注意力分支 ===
        B, T, _ = x.shape
        t_qkv = self.temporal_qkv(x).view(B, T, 3, self.num_heads, self.attn_head_size)
        t_q, t_k, t_v = t_qkv.unbind(2)  # [B, T, H, D/H]

        # 2. 用einsum计算注意力
        t_scores = torch.einsum('bthd,bshd->bhts', t_q, t_k) / math.sqrt(self.attn_head_size)
        t_probs = F.softmax(t_scores, dim=-1)
        t_output = torch.einsum('bhts,bshd->bthd', t_probs, t_v)

        # 3. 合并输出
        t_output = t_output.reshape(B, T, -1)  # [B, T, D]

        return self.dropout(t_output)

class Echo2Det(nn.Module):
    def __init__(self, num_classes=2, sig_len=1024, extend_channel=36, 
                 latent_dim=512, num_heads=8, conv_chan_list=None):
        super().__init__()
        self.conv_chan_list = conv_chan_list or [36,64,128,256]
        
        # 模块维度链式校验 
        self.MTFE = MTFE(extend_channel)
        self.Conv_stem = self._build_conv_stem()
        
        # 时空注意力参数校验
        self.TCA = TCA(
            num_heads=num_heads,
            input_size=self.conv_chan_list[-1],
            hidden_size=latent_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, num_classes),
            nn.Dropout(0.4)
        )

    def _build_conv_stem(self):
        layers = []
        for i in range(len(self.conv_chan_list)-1):
            layers.append(Conv_Block(self.conv_chan_list[i], self.conv_chan_list[i+1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入维度校验 (batch,1,2,1024)
        x = self.MTFE(x)                             # [B,36,1,1024]
        x = self.Conv_stem(x)                        # [B,256,1,1024]
        
        x = x.squeeze(2).permute(0,2,1)              # [B,1024,256]
        # x = x.squeeze(2)              # [B,1024,256]
        x = self.TCA(x)                              # [B,1024,latent_dim]
        return self.classifier(x.mean(1))            # [B,num_classes]

if __name__ == '__main__':
    from thop import clever_format, profile

    x = torch.randn(1, 1, 2, 1024)  #[B,1,2,len]
    model = Echo2Det(latent_dim=512, num_heads=8)
    # print(model)
    print("Output shape:", model(x).shape)  # 应输出[B,2]
    macs, params = profile(model, inputs=(x,), verbose=False)
    flops = macs * 2  # THOP reports MACs; count multiply and add separately for FLOPs.
    macs, flops, params = clever_format([macs, flops, params], "%.3f")
    print(f"MACs/sample: {macs}")
    print(f"FLOPs/sample: {flops}")
    print(f"Params: {params}")
