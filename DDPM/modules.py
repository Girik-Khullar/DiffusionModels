import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# Sinusoidal position embedding for encoding time step
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings based on input timestep indices.
    :param timesteps: Tensor of timestep indices, shape [N].
                      These indices may be fractional.
    :param dim: Dimension of the output embeddings.
    :param max_period: Controls the minimum frequency of the embeddings.
    :return: A Tensor of shape [N x dim] containing positional embeddings.
    """
    half_dim = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
    freqs = freqs.to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# Abstract class for timestep block
class TimestepBlock(nn.Module):
    """
    An abstract module that takes timestep embeddings as input.
    """

    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        raise NotImplementedError


# Sequential model that supports passing timestep embeddings
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to child layers that
    support it as an extra input.
    """

    def forward(self, x, t_emb, c_emb, mask):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)
        return x


# Normalization layer function
def norm_layer(channels):
    """
    Creates a GroupNorm layer with 32 groups.
    :param channels: Number of channels in the input.
    :return: GroupNorm layer.
    """
    return nn.GroupNorm(32, channels)


# Downsample block
class Downsample(nn.Module):
    """
    A downsampling block with optional convolution or average pooling.
    """

    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        return self.op(x)


# Upsample block
class Upsample(nn.Module):
    """
    An upsampling block with optional convolution.
    """

    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# Attention block with shortcut connection
class AttentionBlock(nn.Module):
    """
    An attention block with multiple heads and a residual connection.
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale).softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v).reshape(B, -1, H, W)
        return self.proj(h) + x


# Residual block that accepts timestep and class embeddings
class ResidualBlock(TimestepBlock):
    """
    A residual block that incorporates time and class embeddings.
    """

    def __init__(self, in_channels, out_channels, time_channels, class_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # Embedding layers for time and class
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.class_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # Define shortcut connection
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t, c, mask):
        """
        Forward pass for the residual block.
        :param x: Input tensor with shape [batch_size, in_channels, height, width].
        :param t: Timestep embeddings tensor with shape [batch_size, time_channels].
        :param c: Class embeddings tensor with shape [batch_size, class_channels].
        :param mask: Binary mask with shape [batch_size].
        :return: Output tensor after applying residual block.
        """
        h = self.conv1(x)
        emb_t = self.time_emb(t)
        emb_c = self.class_emb(c) * mask[:, None]
        h += (emb_t[:, :, None, None] + emb_c[:, :, None, None])
        h = self.conv2(h)
        return h + self.shortcut(x)


class UnetModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 model_channels=128,
                 out_channels=3,
                 num_res_blocks=2,
                 attention_resolutions=(8, 16),
                 dropout=0,
                 channel_mult=(1, 2, 2, 2),
                 conv_resample=True,
                 num_heads=4,
                 class_num=10):
        """
        Initialize a U-Net model for conditional image generation with attention mechanisms.

        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        :param model_channels: Base number of model channels in the network.
        :param out_channels: Number of output channels (e.g., 3 for RGB images).
        :param num_res_blocks: Number of residual blocks per down/up sampling level.
        :param attention_resolutions: Resolutions at which to apply attention layers.
        :param dropout: Dropout probability.
        :param channel_mult: Multiplier for channels at each level of the U-Net.
        :param conv_resample: Whether to use convolutional layers for up/downsampling.
        :param num_heads: Number of attention heads.
        :param class_num: Number of class labels for conditional generation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.class_num = class_num

        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_emb = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Class embedding
        class_emb_dim = model_channels
        self.class_emb = nn.Embedding(class_num, class_emb_dim)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, model_channels * mult, time_emb_dim, class_emb_dim, dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_channels.append(ch)
            if level != len(channel_mult) - 1:  # Skip downsampling at the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_channels.append(ch)
                ds *= 2

        # Middle blocks
        self.middle_blocks = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_emb_dim, class_emb_dim, dropout),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_emb_dim, class_emb_dim, dropout)
        )

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult[::-1]):
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(ch + down_block_channels.pop(), model_channels * mult,
                                  time_emb_dim, class_emb_dim, dropout)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level != len(channel_mult) - 1 and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        # Final output layer
        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, timesteps, c, mask):
        """
        Forward pass through the U-Net model.

        :param x: Input tensor with shape [N, C, H, W].
        :param timesteps: 1-D tensor of timesteps.
        :param c: 1-D tensor of class labels.
        :param mask: 1-D tensor indicating conditioned/unconditioned samples.
        :return: Output tensor with shape [N, out_channels, H, W].
        """
        hs = []

        # Embeddings for timestep and class
        t_emb = self.time_emb(timestep_embedding(timesteps, dim=self.model_channels))
        c_emb = self.class_emb(c)

        # Downsampling stage
        h = x
        for module in self.down_blocks:
            h = module(h, t_emb, c_emb, mask)
            hs.append(h)  # Store intermediate results for skip connections

        # Middle stage
        h = self.middle_blocks(h, t_emb, c_emb, mask)

        # Upsampling stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)  # Concatenate with skip connection
            h = module(cat_in, t_emb, c_emb, mask)

        return self.out(h)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    test_N = 64
    test_C = 3
    test_H = 64
    test_W = 64

    model = UnetModel().to(device)

    x = torch.randn(test_N, test_C, test_H, test_W).to(device)
    timesteps = torch.randint(0, 1000, (test_N,)).to(device)
    classes = torch.randint(0, 10, (test_N,)).to(device)
    mask = torch.randn(test_N).to(device)

    print(f"Input tensor shape: {x.shape}")
    print(f"Timesteps tensor shape: {timesteps.shape}")
    print(f"Classes tensor shape: {classes.shape}")
    print(f"Mask tensor shape: {mask.shape}")

    summary(model, input_data=[x, timesteps, classes, mask], depth=3)

    # # net = UNet(device="cpu")
    # net = UNet_conditional(num_classes=10, device="cpu")
    # print(sum([p.numel() for p in net.parameters()]))
    # x = torch.randn(3, 3, 64, 64)
    # t = x.new_tensor([500] * x.shape[0]).long()
    # y = x.new_tensor([1] * x.shape[0]).long()
    # print(net(x, t, y).shape)