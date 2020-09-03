from collections import OrderedDict
import torch
import torch.nn as nn

class DeepHarmonyNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(DeepHarmonyNet,self).__init__()

        features = init_features
        self.encoder1 = DeepHarmonyNet._mainblock(in_channels, features, name = "enc1")
        self.down1 = DeepHarmonyNet._downblock(features, features, name = "down1")
        self.encoder2 = DeepHarmonyNet._mainblock(features , features *2, name = "enc2")
        self.down2 = DeepHarmonyNet._downblock(features *2, features *2, name = "down2")
        self.encoder3 = DeepHarmonyNet._mainblock(features *2, features*4, name = "enc3")
        self.down3 = DeepHarmonyNet._downblock(features*4, features*4, name = "down3")
        self.encoder4 = DeepHarmonyNet._mainblock(features *4, features *8, name = "enc4")
        self.down4 = DeepHarmonyNet._downblock(features *8, features *8, name = "down4")

        self.bottleneck = DeepHarmonyNet._mainblock(features *8, features *16, name = "bottleneck")

        self.up4 = DeepHarmonyNet._upblock(features *16, features *8, name = "up4")
        self.decoder4 = DeepHarmonyNet._mainblock(features *16, features *8, name = "dec4")
        self.up3 = DeepHarmonyNet._upblock(features *8, features *4, name = "up3")
        self.decoder3 = DeepHarmonyNet._mainblock(features *8, features *4, name = "dec3")
        self.up2 = DeepHarmonyNet._upblock(features *4, features *2, name = "up2")
        self.decoder2 = DeepHarmonyNet._mainblock(features *4, features *2, name = "dec2")
        self.up1 = DeepHarmonyNet._upblock(features *2, features, name = "up1")
        self.decoder1 = DeepHarmonyNet._mainblock(features *2, features, name = "dec1")

        self.conv = nn.Conv2d(features+in_channels, in_channels, kernel_size=1)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2( self.down1(enc1) )
        enc3 = self.encoder3( self.down2(enc2))
        enc4 = self.encoder4(  self.down3(enc3))

        bottleneck = self.bottleneck(self.down4(enc4))

        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4,enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3,enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2,enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1,enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = torch.cat((dec1,x), dim=1)
        out = self.conv(out)

        return torch.relu(out)





    @staticmethod
    def _mainblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride =1, 
                            bias=False,
                        ),
                    ),
                    (name + "relu", nn.ReLU(inplace=True)),
                    (name + "norm", nn.BatchNorm2d(num_features=features)),
                   
                ]
            )
        )

    @staticmethod
    def _downblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv_down",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size= 4,
                            stride=2,
                            padding = 1,
                            bias=False,
                        ),
                    ),
                    (name + "relu", nn.ReLU(inplace=True)),                 
                ]
            )
        )

    @staticmethod
    def _upblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv_up",
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=4,
                            stride = 2,
                            padding = 1,
                            bias=False,
                        ),
                    ),
                    (name + "relu", nn.ReLU(inplace=True)),   
                    (name + "norm", nn.BatchNorm2d(num_features=features)),              
                ]
            )
        )


