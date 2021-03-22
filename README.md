# **GAN**oloscopy
```
                                                ▄▄                                                      
  ▄▄█▀▀▀█▄█      ██     ▀███▄   ▀███▀         ▀███                                                      
▄██▀     ▀█     ▄██▄      ███▄    █             ██                                                      
██▀       ▀    ▄█▀██▄     █ ███   █   ▄██▀██▄   ██   ▄██▀██▄ ▄██▀███▄██▀██  ▄██▀██▄▀████████▄▀██▀   ▀██▀
██            ▄█  ▀██     █  ▀██▄ █  ██▀   ▀██  ██  ██▀   ▀████   ▀▀█▀  ██ ██▀   ▀██ ██   ▀██  ██   ▄█  
██▄    ▀████  ████████    █   ▀██▄█  ██     ██  ██  ██     ██▀█████▄█      ██     ██ ██    ██   ██ ▄█   
▀██▄     ██  █▀      ██   █     ███  ██▄   ▄██  ██  ██▄   ▄███▄   ███▄    ▄██▄   ▄██ ██   ▄██    ███    
  ▀▀███████▄███▄   ▄████▄███▄    ██   ▀█████▀ ▄████▄ ▀█████▀ ██████▀█████▀  ▀█████▀  ██████▀     ▄█     
                                                                                     ██        ▄█       
                                                                                   ▄████▄    ██▀        

```

A simple GAN to generate synthetic colorectal images with the associated mask, based on pix2pix architecture


Some examples:

![Example 0](data/yolact_example_0.png)


# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/WalBouss/GANoloscopy.git
   cd GANoloscopy
   ```
 - Set up the environment:
   - Use [Anaconda](https://www.anaconda.com/distribution/) to creat a python environment:
     - Run `conda env create --name myenv`
    - Install [Pytorch](https://pytorch.org/get-started/locally/):
      - Run `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
   - Install dependencies with pip
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install wandb
       pip install opencv-python pillow pycocotools matplotlib 
       ```
# Generate synthetic dataset
To generate a synthetic dataset edit and run the following command (the pretrained weight is already in `./weight` dir):
```Shell
# Specify the lenght and where to save the generated dataset.
python generate_synthetic_dataset.py --dataset_length=100 --path_to_save_img=path/dir/imgs --path_to_save_msk=path/dir/msks

# Use the help option to see a description of all available command line arguments
python generate_synthetic_dataset.py --help
```

# Eval
To eval edit and run the following command
```Shell
# Process a whole folder of masks.
python eval.py --path_to_masks=path/to/masks/ --path_to_save=path/to/save/data --batch_size=10 

# Use the help option to see a description of all available command line arguments
python eval.py --help
```

# Training
To train edit and run the following command
```Shell
# Trains 
python train.py --path_data_train=path_to_training_data --batch_size=10 --epochs=250

# Use the help option to see a description of all available command line arguments
python train.py --help
```

## Logging
GANoloscopy uses [Weights & Biases](https://wandb.ai/home) library to visualize training and validation information by default.

# Custom Datasets
You can also train on your own dataset by following organizing your data as follow:
    .
    ├── dir                    
    │   ├── images          
    │   └── masks                




# Citation
If you use GANoloscopy or this code base in your work, please cite
```
@article{Bousselham-GANoloscopy-2021,
  author  = {Walid Bousselham},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title   = {YOLACT++: Better Real-time Instance Segmentation}, 
  year    = {2021},
}
```



# Contact
For questions about the code, please contact [Walid Bousselham](bousselh@ohsu.edu).
