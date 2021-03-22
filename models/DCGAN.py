import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from data.CustomeDataset import MaskDataset


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        base = 64
        nz = 100
        nc = 3
        self.main = nn.Sequential(
            # Random noise Z of size nz
            nn.ConvTranspose2d(nz, base*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=base*8),
            nn.ReLU(True),
            # --> (base*8) x 4 x 4
            nn.ConvTranspose2d(base * 8, base * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=base*4),
            nn.ReLU(True),
            # --> (base*4) x 8 x 8
            nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=base * 2),
            nn.ReLU(True),
            # --> (base*2) x 16 x 16
            nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=base),
            nn.ReLU(True),
            # --> (base*2) x 32 x 32
            nn.ConvTranspose2d(base, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # --> (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nc = 3
        ndf = 64

        self.main = nn.Sequential(
            # Input (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # --> (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # --> (ndf) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # --> (ndf) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # --> (ndf) x 4 x 4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# Hyperparameters
BatchSize = 512
lr = 5e-4
nz = 100
beta1 = 0.5
epochs = 3000
# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(40),
    transforms.ToTensor(),
])
# DataLoader
# data_path = "../polyps_all_data/images"
data_path = "../polyps_RGB/"
dataset = MaskDataset(data_path, train_transforms)

# # Train DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BatchSize, shuffle=True, num_workers=4, pin_memory=True
)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Criterion
criterion = nn.BCELoss()
# Networks
netG = Generator().to(device)
netD = Discriminator().to(device)
# Optimizers
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
# Set up
fixed_noise = torch.randn(BatchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0
# Tensorboard init
comment = f'GAN: BatchSize={BatchSize}, lr={lr}'
# tb = SummaryWriter(comment=comment)
tb = SummaryWriter(comment=comment, log_dir="./runs_gan1")
# os.path.join("./runs", folder_name)
print("Testtt")
for epoch in range(epochs):
    for batch in train_loader:
        #########################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with real data
        netD.zero_grad()
        real_cpu = batch.to(device)
        # print("real_cpu.size()=", real_cpu.size())
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake data
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)

        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()

        D_G_z2 = output.mean().item()
        optimizerG.step()

    print("epoch: {}/{}".format(epoch, epochs))

    # Tensorboard writer
    tb.add_scalar("Loss/Loss_D", errD, epoch)
    tb.add_scalar("Loss/Loss_G", errG, epoch)
    tb.add_scalar("D(x)", D_G_z1, epoch)
    tb.add_scalar("D(G(z))", D_G_z2, epoch)

    if epoch % 10 == 0:
        fake_samples = fake[:5, :, :, :]
        img_grid_inputs_test = torchvision.utils.make_grid(fake_samples, nrow=5)
        tb.add_image('epoch{}_FakeImg'.format(epoch), img_grid_inputs_test)
tb.flush()


