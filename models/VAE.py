import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class MyVAE(nn.Module):
    def __init__(self, beta=1):
        super(MyVAE, self).__init__()

        base = 16
        self.beta = beta

        self.encoder = nn.Sequential(
            Conv(3, base, 3, stride=2, padding=1),
            Conv(base, 2 * base, 3, padding=1),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),
            Conv(2 * base, 2 * base, 3, padding=1),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),
            Conv(2 * base, 4 * base, 3, padding=1),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),
            Conv(4 * base, 4 * base, 3, padding=1),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),
            nn.Conv2d(4 * base, 64 * base, 8),
            nn.LeakyReLU()
        )
        self.encoder_mu = nn.Conv2d(64 * base, 32 * base, 1)
        self.encoder_logvar = nn.Conv2d(64 * base, 32 * base, 1)

        self.decoder = nn.Sequential(
            nn.Conv2d(32 * base, 64 * base, 1),
            ConvTranspose(64 * base, 4 * base, 8),
            Conv(4 * base, 4 * base, 3, padding=1),
            ConvTranspose(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 4 * base, 3, padding=1),
            ConvTranspose(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 2 * base, 3, padding=1),
            ConvTranspose(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, 2 * base, 3, padding=1),
            ConvTranspose(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, base, 3, padding=1),
            ConvTranspose(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_loss(self, recon_x, x, mu, log_var):
        # bce = F.binary_cross_entropy(recon_x.view(-1, 32 * 32), x.view(-1, 32 * 32), reduction='sum')
        # bce = F.binary_cross_entropy(recon_x.flatten(1), x.flatten(1), reduction='sum')
        # mse = torch.nn.MSELoss()(recon_x.view(x.size()), x)
        mse = torch.nn.MSELoss()(recon_x, x)
        # mse = F.l1_loss(recon_x, x)

        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # divide by batch size
        return mse + self.beta*kld, mse, kld


class MyVAEnMsk(nn.Module):
    def __init__(self, beta=1):
        super(MyVAEnMsk, self).__init__()

        base = 32
        self.beta = beta

        self.encoder = nn.Sequential(
            Conv(4, base, 3, stride=2, padding=1),
            Conv(base, 2 * base, 3, padding=1),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),
            Conv(2 * base, 2 * base, 3, padding=1),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),
            Conv(2 * base, 4 * base, 3, padding=1),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),
            Conv(4 * base, 4 * base, 3, padding=1),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),
            nn.Conv2d(4 * base, 64 * base, 8),
            nn.LeakyReLU()
        )
        self.encoder_mu = nn.Conv2d(64 * base, 32 * base, 1)
        self.encoder_logvar = nn.Conv2d(64 * base, 32 * base, 1)

        self.decoder = nn.Sequential(
            nn.Conv2d(32 * base, 64 * base, 1),
            ConvTranspose(64 * base, 4 * base, 8),
            Conv(4 * base, 4 * base, 3, padding=1),
            ConvTranspose(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 4 * base, 3, padding=1),
            ConvTranspose(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 2 * base, 3, padding=1),
            ConvTranspose(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, 2 * base, 3, padding=1),
            ConvTranspose(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, base, 3, padding=1),
            ConvTranspose(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 4, 3, padding=1),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # img_n_msk = self.decode(z)
        # return img_n_msk[:, :3, :, :], img_n_msk[:, 4, :, :], mu, logvar
        return self.decode(z), mu, logvar

    def get_loss(self, recon_x, x, mu, log_var):
        # bce = F.binary_cross_entropy(recon_x.view(-1, 32 * 32), x.view(-1, 32 * 32), reduction='sum')
        # bce = F.binary_cross_entropy(recon_x.flatten(1), x.flatten(1), reduction='sum')
        # mse = torch.nn.MSELoss()(recon_x.view(x.size()), x)
        mse = torch.nn.MSELoss()(recon_x[:, :3, :, :], x[:, :3, :, :])
        binary_loss = F.binary_cross_entropy(recon_x[:, 3, :, :], x[:, 3, :, :])
        # mse = F.l1_loss(recon_x, x)

        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # divide by batch size
        return mse + self.beta*kld + binary_loss, mse, kld


class CustomeConvAE1(nn.Module):
    def __init__(self, z_dim=10):
        super(CustomeConvAE1, self).__init__()
        self.name = "CustomeConvAE1"
        self.criterion = nn.MSELoss()

        # Encoder
        self.encoder = nn.Sequential(
        #     Conv2DReLU(in_channels=3, out_channels=16, kernel_size=4, padding=1, stride=2),
        #     nn.MaxPool2d(2, 2),
        #     Conv2DReLU(in_channels=16, out_channels=32, kernel_size=4, padding=1, stride=2),
        #     Conv2DReLU(in_channels=16, out_channels=4, kernel_size=4, padding=1, stride=2),
        #     nn.MaxPool2d(2, 2),
            Conv2DReLU(3, 32, kernel_size=4, stride=2, padding=1),  # (128,128)  # (B, nc, 32, 32) -> (B, 32, 16, 16)
            Conv2DReLU(32, 64, kernel_size=4, stride=2, padding=1),  # (64,64) # (B, 32, 32, 32) -> (B, 64, 8, 8)
            Conv2DReLU(64, 128, kernel_size=4, stride=2, padding=1),  # (32,32)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(128, 256, kernel_size=4, stride=2, padding=1),  # (16,16)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(256, 512, kernel_size=4, stride=2, padding=1),  # (8,8)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(512, 1024, kernel_size=4, stride=2, padding=1),  # (4,4)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(1024, 2048, kernel_size=4, stride=1),  # (  # (B, 128, 4, 4) -> (B, 256, 1, 1)
            Conv2DReLU(2048, z_dim * 2, kernel_size=1, stride=1), # (  # (B, 128, 4, 4) -> (B, 256, 1, 1)

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim*2, 2048, 4),  # (B, 2048, 1, 1) -> (B, 1024, 4, 4)
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),  # (B, 2048, 1, 1) -> (B, 1024, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # (B, 1024, 1, 1) -> (B, 512, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # (B, 256, 1, 1) -> (B, 64, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 4, 2, 1),  # (B, 256, 1, 1) -> (B, 64, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 64, 4, 4) -> (B, 64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # (B, 64, 8, 8) -> (B, 32, 16, 16)
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 3, 4, 2, 1),  # (B, 32, 16, 16) -> (B, nc, 32, 32),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def get_loss(self, reconstructed, x):
        return self.criterion(reconstructed, x)


class CustomeConvAE(nn.Module):
    def __init__(self):
        super(CustomeConvAE, self).__init__()
        self.name = "ConvAE"
        self.criterion = nn.MSELoss()

        # Encoder
        self.encoder = nn.Sequential(
            Conv2DReLU(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            Conv2DReLU(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            Conv2DReLU(in_channels=16, out_channels=4, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def get_loss(self, reconstructed, x):
        return self.criterion(reconstructed, x)


class VanillaSimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "VanillaSimpleAE"
        self.criterion = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=16384, out_features=8192
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=8192, out_features=4096
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=4096, out_features=2048
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2048, out_features=1024
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=784, out_features=512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=512, out_features=256
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256, out_features=128
            ),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=128, out_features=256
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256, out_features=512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=512, out_features=784
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x.view(-1, 784))
        reconstructed = self.decoder(encoded)
        return reconstructed

    def get_loss(self, reconstructed, x):
        return self.criterion(reconstructed.view(x.size()).squeeze(), x.squeeze())


class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SimpleAE"
        self.criterion = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=784, out_features=512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=512, out_features=256
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256, out_features=128
            ),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=128, out_features=256
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256, out_features=512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=512, out_features=784
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x.view(-1, 784))
        reconstructed = self.decoder(encoded)
        return reconstructed

    def get_loss(self, reconstructed, x):
        return self.criterion(reconstructed.view(x.size()).squeeze(), x.squeeze())


class Conv2DReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilatation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilatation)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.name = "ConvAE"
        self.criterion = nn.MSELoss()

        # Encoder
        self.encoder = nn.Sequential(
            Conv2DReLU(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            Conv2DReLU(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            Conv2DReLU(in_channels=16, out_channels=4, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def get_loss(self, reconstructed, x):
        return self.criterion(reconstructed.squeeze(), x.squeeze())


class ConvAEFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "ConvAEFC"
        self.mse = nn.MSELoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        # Encoder
        self.encoder = nn.Sequential(
            Conv2DReLU(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            Conv2DReLU(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            Conv2DReLU(in_channels=16, out_channels=4, kernel_size=3, padding=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * 7 * 7),
            nn.ReLU()
        )
        self.classif = nn.Sequential(
            nn.Linear(4 * 7 * 7, 4 * 7 * 7),
            nn.ReLU(),
            nn.Linear(4 * 7 * 7, 10),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        fc = self.fc(encoded.view(-1, 4 * 7 * 7))
        classif = self.classif(fc)
        reconstructed = self.decoder(fc.view(-1, 4, 7, 7))
        return reconstructed, classif

    def get_loss(self, reconstructed, x, classif, gt):
        return self.mse(reconstructed.squeeze(), x.squeeze()) + self.cross_entropy(classif, gt)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.name = "VAE"

        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256, 10)
        self.logvar = nn.Linear(256, 10)

        self.decoder = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decode(z), mu, log_var

    def get_loss(self, recon_x, x, mu, log_var):
        bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        mse = torch.nn.MSELoss()(recon_x.view(x.size()), x)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return mse + kld + bce


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BVAE(nn.Module):
    def __init__(self, beta=None, z_dim=10, nc=1):
        super(BVAE, self).__init__()
        self.name = "BVAE"
        if beta is None:
            self.beta = 1
        else:
            self.beta = beta
        self.z_dim = z_dim

        self.encoder = nn.Sequential(

            Conv2DReLU(nc, 32, kernel_size=4, stride=2, padding=1),# (128,128)  # (B, nc, 32, 32) -> (B, 32, 16, 16)
            Conv2DReLU(32, 64, kernel_size=4, stride=2, padding=1),# (64,64) # (B, 32, 32, 32) -> (B, 64, 8, 8)
            Conv2DReLU(64, 128, kernel_size=4, stride=2, padding=1),# (32,32)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(128, 256, kernel_size=4, stride=2, padding=1),# (16,16)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(256, 512, kernel_size=4, stride=2, padding=1),# (8,8)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(512, 1024, kernel_size=4, stride=2, padding=1),# (4,4)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(1024, 2048, kernel_size=4, stride=1),#(  # (B, 128, 4, 4) -> (B, 256, 1, 1)
            View((-1, 2048)),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, z_dim * 2)
        )
        self.weight_init()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            View((-1, 2048, 1, 1)),
            nn.ConvTranspose2d(2048, 1024, 4),  # (B, 2048, 1, 1) -> (B, 1024, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # (B, 1024, 1, 1) -> (B, 512, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # (B, 256, 1, 1) -> (B, 64, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 4, 2, 1),  # (B, 256, 1, 1) -> (B, 64, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # (B, 64, 4, 4) -> (B, 64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 64, 8, 8) -> (B, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # (B, 32, 16, 16) -> (B, nc, 32, 32)
            nn.Sigmoid()
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def encode(self, x):
        return self.encoder(x)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # print(x.size())
        encoded = self.encode(x)
        # print(encoded.size())
        mu, log_var = encoded[:, :self.z_dim], encoded[:, self.z_dim:]
        z = self.sampling(mu, log_var)
        x_recon = self.decode(z)
        # print(x_recon.size())
        return x_recon, mu, log_var

    def get_loss(self, recon_x, x, mu, log_var):
        # bce = F.binary_cross_entropy(recon_x.view(-1, 32 * 32), x.view(-1, 32 * 32), reduction='sum')
        # bce = F.binary_cross_entropy(recon_x.flatten(1), x.flatten(1), reduction='sum')
        # mse = torch.nn.MSELoss()(recon_x.view(x.size()), x)
        mse = torch.nn.MSELoss()(recon_x, x)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # divide by batch size
        return mse + kld * self.beta, mse, kld


class ConvBVAE(nn.Module):
    def __init__(self, beta=None, z_dim=10, nc=1):
        super(ConvBVAE, self).__init__()
        self.name = "ConvBVAE"
        if beta is None:
            self.beta = 1
        else:
            self.beta = beta
        self.z_dim = z_dim

        self.encoder = nn.Sequential(

            Conv2DReLU(nc, 32, kernel_size=4, stride=2, padding=1),# (128,128)  # (B, nc, 32, 32) -> (B, 32, 16, 16)
            Conv2DReLU(32, 64, kernel_size=4, stride=2, padding=1),# (64,64) # (B, 32, 32, 32) -> (B, 64, 8, 8)
            Conv2DReLU(64, 128, kernel_size=4, stride=2, padding=1),# (32,32)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(128, 256, kernel_size=4, stride=2, padding=1),# (16,16)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(256, 512, kernel_size=4, stride=2, padding=1),# (8,8)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(512, 1024, kernel_size=4, stride=2, padding=1),# (4,4)  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(1024, 2048, kernel_size=4, stride=1),#(  # (B, 128, 4, 4) -> (B, 256, 1, 1)
            Conv2DReLU(2048, z_dim * 2, kernel_size=1, stride=1),#(  # (B, 128, 4, 4) -> (B, 256, 1, 1)
            )
        self.weight_init()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            View((-1, 2048, 1, 1)),
            nn.ConvTranspose2d(2048, 1024, 4),  # (B, 2048, 1, 1) -> (B, 1024, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # (B, 1024, 1, 1) -> (B, 512, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # (B, 256, 1, 1) -> (B, 64, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 4, 2, 1),  # (B, 256, 1, 1) -> (B, 64, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # (B, 64, 4, 4) -> (B, 64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 64, 8, 8) -> (B, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # (B, 32, 16, 16) -> (B, nc, 32, 32)
            nn.ReLU() # nn.Sigmoid()
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def encode(self, x):
        return self.encoder(x)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # print(x.size())
        encoded = self.encode(x)
        # print(encoded.size())
        mu, log_var = encoded[:, :self.z_dim], encoded[:, self.z_dim:]
        z = self.sampling(mu, log_var)
        x_recon = self.decode(z)
        # print(x_recon.size())
        return x_recon, mu, log_var

    def get_loss(self, recon_x, x, mu, log_var):
        # bce = F.binary_cross_entropy(recon_x.view(-1, 32 * 32), x.view(-1, 32 * 32), reduction='sum')
        # bce = F.binary_cross_entropy(recon_x.flatten(1), x.flatten(1), reduction='sum')
        # mse = torch.nn.MSELoss()(recon_x.view(x.size()), x)
        mse = torch.nn.MSELoss()(recon_x, x)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # divide by batch size
        return mse + kld * self.beta

