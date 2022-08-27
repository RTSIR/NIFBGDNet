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
