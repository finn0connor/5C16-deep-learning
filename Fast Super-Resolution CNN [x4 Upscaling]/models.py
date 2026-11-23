import torch                                      # The main PyTorch library
import torch.nn as nn                             # Contains pytorch network building blocks (e.g., layers)
import torch.nn.functional as F                   # Contains functions for neural network operations (e.g., activation functions)


class SuperResolutionModel(nn.Module):
    # x4 super-resolution: (B, 3, 32, 32) -> (B, 3, 128, 128), pixel values in [0, 1]
    #
    # output = bicubic upscale of the input + detail learned by a small CNN
    # (the network only learns the missing detail - residual learning, as in VDSR, Kim et al. 2016)
    #
    # The 3x3 convolutions run on the small 32x32 image (cheap), and two transposed
    # convolutions (Handout 06) enlarge the features 32 -> 64 -> 128.

    def __init__(self, channels=64):
        # Define the layers you want to use here
        super().__init__()

        # feature extraction at 32x32 (padding='same' keeps the size, as in lab 5)
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')

        # upsampling: each transposed conv doubles the size
        # output = (input - 1) * stride - 2 * padding + kernel = (32 - 1) * 2 - 2 + 4 = 64
        self.up1 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

        # back to 3 colour channels (no ReLU: the detail can be negative)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding='same')

        # start the last layer with tiny weights, so the untrained model outputs roughly
        # the bicubic image and training starts from a good baseline
        nn.init.normal_(self.conv_out.weight, std=0.001)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        # Define the forward pass of your network here
        upscaled = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.up1(out))
        out = F.relu(self.up2(out))
        detail = self.conv_out(out)

        output = upscaled + detail
        if not self.training:
            output = torch.clamp(output, 0, 1)    # keep outputs in [0, 1] when evaluating
        return output