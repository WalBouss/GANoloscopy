import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from data.CustomeDataset import GenerateMaskDataset
from models.pix2pix import UNetGenerator


parser = argparse.ArgumentParser(description="GANoloscopy UI Script to generate synthetic dataset")
parser.add_argument('--batch_size', default=10, type=int,
                    help='Batch size for training')
parser.add_argument('--dataset_length', default=10, type=int,
                    help='Number of images/masks to generate')
parser.add_argument('--input_size', default=256, type=int,
                    help='Input size of the network')
parser.add_argument('--path_to_save_img', default="./results/img", type=str,
                    help='Path to save the generated images.')
parser.add_argument('--path_to_save_msk', default="./results/msk", type=str,
                    help='Path to save the generated masks.')
parser.add_argument('--path_to_model', default="./weights/GANoloscopy_trained.pt", type=str,
                    help='Path to trained model.')
args = parser.parse_args()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.ToTensor(),
])
# Dataset
# # Generate random segmentation masks
mask_generated_dataset = GenerateMaskDataset(dataset_length=args.dataset_length, transform=train_transforms)
mask_loader = torch.utils.data.DataLoader(dataset=mask_generated_dataset,
                                          batch_size=args.batch_size)

# Load Trained Model
netG = UNetGenerator(in_channel=1, out_channel=3)
netG.load_state_dict(torch.load(args.path_to_model, map_location=torch.device('cpu')))
netG.to(device)

# Eval on data
# not eval mode to keep batchnorm and dropout ->with torch.no_grad()
n = 0
with torch.no_grad():
    for masks in mask_loader:
        masks = masks.to(device)

        rgb_reconstructed = netG(masks)
        for i in range(len(rgb_reconstructed)):
            image = rgb_reconstructed[i].numpy()
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(np.transpose(image, (1,2,0)))
            axs[1].imshow(np.transpose(masks[i], (1,2,0)))
            # plt.imshow(np.transpose(image, (1,2,0)))
            plt.show(block=False)
            t = input("Keep this images?  ")
            print(t)
            if t == 'n' or t == 'no':
                print("Ok let put it in the bin!!!!")
                pass
            elif t == 'y' or t == "" or t == "yes":
                print("Let save it!")
                save_image(rgb_reconstructed[i], os.path.join(args.path_to_save_img, f"synthetic_{n}.png"))
                save_image(masks[i], os.path.join(args.path_to_save_msk, f"synthetic_{n}.png"))
                n += 1
            elif t == 'q':
                break
            else:
                print("you have to make a choice!")

            plt.close()
            break



