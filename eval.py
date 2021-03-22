import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
import argparse


from data.CustomeDataset import MaskDataset
from models.pix2pix import UNetGenerator

parser = argparse.ArgumentParser(description="GANoloscopy Testing Script")
parser.add_argument('--batch_size', default=10, type=int,
                    help='Batch size for training')
parser.add_argument('--input_size', default=256, type=int,
                    help='Input size of the network')
parser.add_argument('--path_to_masks', default="./data/test/images", type=str,
                    help='Path to testing masks.')
parser.add_argument('--path_to_save', default="./results", type=str,
                    help='Path to save the generated images.')
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
# # Test path
test_msk = args.path_to_masks
dataset = MaskDataset(test_msk, transform=train_transforms)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
)
# Load Trained Model
netG = UNetGenerator(in_channel=1, out_channel=3)
netG.load_state_dict(torch.load(args.path_to_model, map_location=torch.device('cpu')))
netG.to(device)

# Eval on data
# not eval mode to keep batchnorm and dropout ->with torch.no_grad()
n = 0
with torch.no_grad():
    for batch in test_loader:
        masks = batch.to(device)

        rgb_reconstructed = netG(masks)
        for i in range(len(rgb_reconstructed)):
            save_image(rgb_reconstructed[i], os.path.join(args.path_to_save , f"GANoloscopy_{n}.png")) #TODO: give better name to generated files
            n += 1
            print("images saved!")