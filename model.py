"""
This module defines a convolutional neural network (CNN) architecture for image classification.

The model implements the following architecture:
    - Channels: 3 => 32
    - 1 block: Conv(3x3, same) => BN => ReLU => MaxPool(2x2, s=2)
    - Head: GAP => FC(1) => Sigmoid
    
"""

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer for conv1 output.
        relu1 (nn.ReLU): ReLU activation function.
        pool1 (nn.MaxPool2d): Max pooling layer.
        global_pool (nn.AdaptiveAvgPool2d): Global average pooling to reduce spatial dimensions.
        fc_out (nn.Linear): output layer for classification.
    """

    def __init__(self, args) -> None:
        """
        Initialize the CNN model.

        Args:
            args: Object containing configuration parameters such as dataset directories and image settings.
        """
        super(CNNModel, self).__init__()

        # Convolutional block
        self.conv1 = nn.Conv2d(
            in_channels=args.channel,
            out_channels=32,
            kernel_size=3,
            padding="same",
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc_out = nn.Linear(32, args.num_classes)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes),
            representing raw, unnormalized class logits.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x
