import torch
import torch.nn as nn

# Attention Gate:
class AttentionGate(nn.Module):
    """
    Attention Gate (AG) filters the skip connections to focus on relevant features .
    It suppresses irrelevant background regions and highlights target objects.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # W_g: Linear transformation for the gating signal (from the deeper decoder layer)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # W_x: Linear transformation for the skip connection (from the encoder layer)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        #psi: Calculate the Attention coefficients alpha (0 to 1)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # Force the value in range (0, 1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal, x: skip connection feature map
        g1 = self.W_g(g) # Transform gating signal
        x1 = self.W_x(x) # Transform skip connection
        psi = self.relu(g1+x1) # Combine both signals to find feature overlap
        psi = self.psi(psi) # Generate the attention mask (0 - 1)
        return x * psi # Rescale the skip connection by attention weights

class ConvBlock(nn.Module):
    """
    Double Convolutional Block: (Conv -> BatchNorm -> ReLU) * 2
    """
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    """
    Attention U-Net implementation for multi-spectral satellite imagery.
    Default input channels: 4 (Red, Green, BLue, NIR)
    """
    def __init__(self, img_ch=4, output_ch=1):
        super(AttentionUNet, self).__init__()

        # Maxpooling to reduce spatial dimensions by half
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ENCODER:Down-sampling path to extract multi-scale features
        self.Conv1 = ConvBlock(img_ch, 64) # Level 1: 256x256
        self.Conv2 = ConvBlock(64, 128) # Level 2: 128 x128
        self.Conv3 = ConvBlock(128, 256)  # Level 3: 64x64
        self.Conv4 = ConvBlock(256, 512) # Level 4: 32x32
        self.Conv5 = ConvBlock(512, 1024) # Level 5 (Bottleneck) 16x16

        # DECODER: Up-sampling path with Attention Gates to reconstruct tha mask
        # Level 5 & 4
        self.Up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.Att5 = AttentionGate(F_g = 1024, F_l=512, F_int=256) # Refine level 4 features
        self.Up_conv5 = ConvBlock(1024 + 512, 512) # Concatenate decoded + attended features

        # Level 4 to 3
        self.Up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.Att4 = AttentionGate(F_g=512, F_l=256,F_int=128) # Refine level 3 features
        self.Up_conv4 = ConvBlock(512 + 256, 256)

        # Level 3 to 2
        self.Up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.Att3 = AttentionGate(F_g=256, F_l=128, F_int=64) # Refine level 2 features
        self.Up_conv3 = ConvBlock(256+128, 128)

        # Level 2 to 1
        self.Up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.Att2 = AttentionGate(F_g =128, F_l=64, F_int=32) # Refine level 1 features
        self.Up_conv2 = ConvBlock(128+64, 64)

        # Final output layer: 1x1 convolution to product binary forest mask
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # --- ENCODING PATH----
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5) # Bottleneck

        # --- DECODING PATH with Attention
        d5 = self.Up5(e5) # Upsample bottle neck
        x4 = self.Att5(g=d5, x=e4)       # Attend to encoder layer 4
        d5 = torch.cat((x4, d5), dim=1)  # Concatenate refined features
        d5 = self.Up_conv5(d5)           # Double conv

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # Final output mapping
        out = self.Conv_1x1(d2)
        return out
