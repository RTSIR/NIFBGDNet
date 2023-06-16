# NIFBGDNet
This repo contains the KERAS implementation of "Multi Scale Pixel Attention and Feature Extraction based Neural Network for Image Denoising"


# Run Experiments

To test for blind gray denoising using NIFBGDNet write:

python Test_gray.py

The resultant images will be stored in 'Test_Results/Gray/'

To test for blind color denoising using NIFBGDNet write:

python Test_color.py

The resultant images will be stored in 'Test_Results/Color/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database is also displayed in the console as output.

# Train NIFBGDNet gray denoising network

To train the NIFBGDNet gray denoising network, first download the [BSD400 dataset](https://github.com/smartboy110/denoising-datasets/tree/main/BSD400) and save this dataset inside the main folder of this project. Then generate the training data using:

python Generate_Patches_Gray.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the NIFBGDNet model file using:

python NIFBGDNet_Gray.py

This will save the 'NIFBGDNet_Gray.h5' file in the folder 'Pretrained_models/'.


# Train NIFBGDNet color denoising network

To train the NIFBGDNet color denoising network, first download the [CBSD432 dataset](https://github.com/Magauiya/Extended_SURE/tree/master/Dataset/CBSD432) and save this dataset inside the main folder of this project. Then generate the training data using:

python Generate_Patches_Color.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the NIFBGDNet model file using:

python NIFBGDNet_Color.py

This will save the 'NIFBGDNet_Color.h5' file in the folder 'Pretrained_models/'.

# Citation
@article{thakur2023multi,
  title={Multi scale pixel attention and feature extraction based neural network for image denoising},
  author={Thakur, Ramesh Kumar and Maji, Suman Kumar},
  journal={Pattern Recognition},
  volume={141},
  pages={109603},
  year={2023},
  publisher={Elsevier}
}
