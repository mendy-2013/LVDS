import math
import torch
import torch.nn as nn

class CBAMBlock(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock, self).__init__()
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x

if __name__ == "__main__":
    feature_maps = torch.randn((8, 54, 32, 32))
    model = CBAMBlock("FC", 5, channels = 54, ratio = 9)
    out=model(feature_maps)
    print(out.size())
    model = CBAMBlock("Conv", 5, channels = 54, gamma = 2, b = 1)
    out=model(feature_maps)
    print(out.size())
