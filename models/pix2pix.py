import torch
import torch.nn as nn
from functools import partial


#========================= Discriminator =========================
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

#========================= Generator =========================

def encoder_block(in_channel, out_channel, is_batchnorm=True, kernel_size=4, stride=2, padding=1):
    layers = [nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                        padding=padding)]
    if is_batchnorm:
        layers.append(nn.BatchNorm2d(out_channel))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


def decoder_block(in_channel, out_channel, is_dropout=True, kernel_size=4, stride=2, padding=1):
    layers = [nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,
                                 kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm2d(out_channel)]
    if is_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# TODO add adaptive kernel_size to UNetGenerator
class UNetGenerator(nn.Module):
    def __init__(self, in_channel=1, out_channel=3, kernel_size=4):
        """Construct a PatchGAN discriminator

                Parameters:
                    input_nc (int)  -- the number of channels in input images
                    ndf (int)       -- the number of filters in the last conv layer
                    n_layers (int)  -- the number of conv layers in the discriminator
                    norm_layer      -- normalization layer
                """
        super(UNetGenerator, self).__init__()

        # Encoder ->128x128
        self.encode1 = encoder_block(in_channel=in_channel, out_channel=64, is_batchnorm=False) #->64x64
        self.encode2 = encoder_block(64, 128, is_batchnorm=True)  # ->32x32
        self.encode3 = encoder_block(128, 256, is_batchnorm=True)  # ->16x16
        self.encode4 = encoder_block(256, 512, is_batchnorm=True)  # ->8x8
        self.encode5 = encoder_block(512, 512, is_batchnorm=True)  # ->4x4
        self.encode6 = encoder_block(512, 512, is_batchnorm=True)  # ->2x2
        self.encode7 = encoder_block(512, 512, is_batchnorm=True)  # ->1x1
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(),
        )
        # Decoder
        self.decode7 = decoder_block(in_channel=512, out_channel=512, is_dropout=True)
        self.decode6 = decoder_block(in_channel=1024, out_channel=512, is_dropout=True)
        self.decode5 = decoder_block(in_channel=1024, out_channel=512, is_dropout=True)
        self.decode4 = decoder_block(in_channel=1024, out_channel=512, is_dropout=False)
        self.decode3 = decoder_block(in_channel=1024, out_channel=256, is_dropout=False)
        self.decode2 = decoder_block(in_channel=512, out_channel=128, is_dropout=False)
        self.decode1 = decoder_block(in_channel=256, out_channel=64, is_dropout=False)
        # Output
        self.output = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=out_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        return

    def skip_connection(self, upsampled, bypass):
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)
        e6 = self.encode6(e5)
        e7 = self.encode7(e6)
        # Bottleneck
        b = self.bottleneck(e7)
        # Decode
        upspl7 = self.decode7(b)
        d7 = self.skip_connection(upspl7, e7)
        upspl6 = self.decode6(d7)
        d6 = self.skip_connection(upspl6, e6)
        upspl5 = self.decode5(d6)
        d5 = self.skip_connection(upspl5, e5)
        upspl4 = self.decode4(d5)
        d4 = self.skip_connection(upspl4, e4)
        upspl3 = self.decode3(d4)
        d3 = self.skip_connection(upspl3, e3)
        upspl2 = self.decode2(d3)
        d2 = self.skip_connection(upspl2, e2)
        upspl1 = self.decode1(d2)
        d1 = self.skip_connection(upspl1, e1)
        # Output
        output = self.output(d1)
        return output
