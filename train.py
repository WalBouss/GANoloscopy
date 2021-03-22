import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

import os
import argparse
import wandb
from data.CustomeDataset import PolypNMaskDataset
from models.pix2pix import UNetGenerator, NLayerDiscriminator

#                                                 ▄▄
#   ▄▄█▀▀▀█▄█      ██     ▀███▄   ▀███▀         ▀███
# ▄██▀     ▀█     ▄██▄      ███▄    █             ██
# ██▀       ▀    ▄█▀██▄     █ ███   █   ▄██▀██▄   ██   ▄██▀██▄ ▄██▀███▄██▀██  ▄██▀██▄▀████████▄▀██▀   ▀██▀
# ██            ▄█  ▀██     █  ▀██▄ █  ██▀   ▀██  ██  ██▀   ▀████   ▀▀█▀  ██ ██▀   ▀██ ██   ▀██  ██   ▄█
# ██▄    ▀████  ████████    █   ▀██▄█  ██     ██  ██  ██     ██▀█████▄█      ██     ██ ██    ██   ██ ▄█
# ▀██▄     ██  █▀      ██   █     ███  ██▄   ▄██  ██  ██▄   ▄███▄   ███▄    ▄██▄   ▄██ ██   ▄██    ███
#   ▀▀███████▄███▄   ▄████▄███▄    ██   ▀█████▀ ▄████▄ ▀█████▀ ██████▀█████▀  ▀█████▀  ██████▀     ▄█
#                                                                                      ██        ▄█
#                                                                                    ▄████▄    ██▀


parser = argparse.ArgumentParser(description="GANoloscopy Training Script")
parser.add_argument('--batch_size', default=10, type=int,
                    help='Batch size for training')
parser.add_argument('--lr', '--learning_rate', default=0.0003, type=float,
                    help='Initial learning rate.')
parser.add_argument('--lambda_L1', default=10, type=int,
                    help='Coefficient to scale the emphasis to put on L1 loss in the total loss.')
parser.add_argument('--epochs', default=200, type=int,
                    help='Max number of epochs.')
parser.add_argument('--beta1', default=0.5, type=float,
                    help='beta1 parameter for Adam optimizer.')
parser.add_argument('--input_size', default=256, type=int,
                    help='Input size of the network')
parser.add_argument('--path_data_train', default="./data/train", type=str,
                    help='Path to training images.')
parser.add_argument('--path_data_val', default="./data/val", type=str,
                    help='Path to validation images.')

args = parser.parse_args()

print("Set up...")
# W&B set up
HPP_DEFAULT = dict(
    model="GANoloscopy",
    lr=args.lr,
    batch_size=args.batch_size,
    epochs=args.epochs,
    input_size=args.input_size,
    beta1=args.beta1,
    lambda_L1=args.lambda_L1,
)


run = wandb.init(project="GANoloscopy", config=HPP_DEFAULT)
config = wandb.config
# config = HPP_DEFAULT
print(config)
print(config.lr)
# print(run.name)
save_model_dir = f'GANoloscopy_{run.name}'
save_model_dir = os.path.join("./models", save_model_dir)

# Criterion
criterion_L1 = nn.L1Loss()
criterion_cGAN = nn.BCEWithLogitsLoss()

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((config.input_size, config.input_size)),
    transforms.ToTensor(),
])
# Dataset
# # Train paths
training_data = args.path_data_train
training_img = os.path.join(training_data, "images")
training_msk = os.path.join(training_data, "masks")

dataset = PolypNMaskDataset(training_img, training_msk, transform=train_transforms)

# # Train DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
)
# # Validation paths
val_data = args.path_data_val
val_img = os.path.join(val_data, "images")
val_msk = os.path.join(val_data, "masks")

dataset_val = PolypNMaskDataset(val_img, val_msk, transform=train_transforms)

# # Validation DataLoader
val_loader = torch.utils.data.DataLoader(
    dataset_val, batch_size=10, shuffle=True, num_workers=4, pin_memory=True
)
# # To load the same images to W&B every 10 epochs
val_batch = next(iter(val_loader))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Networks
netG = UNetGenerator(in_channel=1, out_channel=3).to(device)
netD = NLayerDiscriminator(input_nc=4).to(device)

wandb.watch(netD)
wandb.watch(netG)

# Optimizers
optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
# Set up
real_label = torch.tensor(1.0, dtype=torch.float32, device=device)
fake_label = torch.tensor(0.0, dtype=torch.float32, device=device)


print("Training ...")
for epoch in range(config.epochs):
    netD.train()
    netG.train()

    g_loss_train = 0
    d_real_loss_train = 0
    d_fake_loss_train = 0
    d_loss_train = 0

    for batch in train_loader:
        #########################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z,x)))
        ###########################
        # Train with real data
        netD.zero_grad()
        images, masks = batch
        images, masks = images.to(device), masks.to(device)
        imagesG, masksG = images, masks # TODO: change this

        input_D = torch.cat((images, masks), dim=1)
        output_D = netD(input_D)
        label = real_label.expand_as(output_D) # Create target label for real data

        errD_real = criterion_cGAN(output_D, label)

        # Train with fake data
        rgb_reconstructed = netG(masks)

        input_fake_D = torch.cat((rgb_reconstructed, masks), dim=1)
        output_fake_D = netD(input_fake_D)

        label = fake_label.expand_as(output_fake_D) # Create target label for fake data
        errD_fake = criterion_cGAN(output_fake_D, label)

        errD = (errD_real + errD_fake) * 0.5
        errD.backward()

        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z,x)))
        ###########################
        netG.zero_grad()
        rgb_reconstructed = netG(masksG)
        input_fake_D = torch.cat((rgb_reconstructed, masksG), dim=1)
        output_fake_D = netD(input_fake_D)

        label = real_label.expand_as(output_fake_D)
        errG = criterion_cGAN(output_fake_D, label) + config.lambda_L1 * criterion_L1(rgb_reconstructed, imagesG)
        errG.backward()
        optimizerG.step()

        g_loss_train += errG
        d_real_loss_train += errD_real
        d_fake_loss_train += errD_fake
        d_loss_train = errD

    g_loss_train /= len(dataset)
    d_real_loss_train /= len(dataset)
    d_fake_loss_train /= len(dataset)
    d_loss_train /= len(dataset)

    # Log metrics to W&B
    wandb.log({
        'g_loss_train': g_loss_train,
        'd_real_loss_train': d_real_loss_train,
        'd_fake_loss_train': d_fake_loss_train,
        'd_loss_train': d_loss_train
    })

    print("epoch: {}/{}".format(epoch, config.epochs))

    if epoch % 10 == 0:
        # netG.eval() --> otherwise dropout and batchnorm disabled in eval mode
        print("saving model ...")
        model_name = os.path.join(save_model_dir, f"GANoloscopy_{epoch}.pt")
        torch.save(netG.state_dict(), model_name)
        wandb.save(model_name)
        print("model saved!")
        with torch.no_grad():
            images, masks = val_batch
            images, masks = images.to(device), masks.to(device)

            rgb_reconstructed = netG(masks).detach()
            wandb.log({
                'reconstructed': [wandb.Image(i) for i in rgb_reconstructed],
                'target': [wandb.Image(i) for i in images],
                'masks': [wandb.Image(i) for i in masks],
            })
