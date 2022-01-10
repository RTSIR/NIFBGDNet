# MFEBDN
This repo contains the KERAS implementation of "Blind Gaussian Deep Denoiser Network using Multi-Scale Feature Extraction(MFEBDN)"


# Run Experiments

To test for blind gray denoising using MFEBDN write:

python Test_gray.py

The resultant images will be stored in 'Test_Results/Gray/'

To test for blind color denoising using MFEBDN write:

python Test_color.py

The resultant images will be stored in 'Test_Results/Color/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database is also displayed in the console as output.


# Train MFEBDN gray denoising network

To train for blind gray denoising using MFEBDN, first generate the training data using:

python generateData.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the MFEBDN_Gray model file using:

python MFEBDN_Gray.py

This will save the 'MFEBDN_Gray.h5' file of in the folder 'Pretrained_models/'.


# Train MFEBDN color denoising network

To train for blind color denoising using MFEBDN, first generate the training data using:

python generate_patches_rgb_blind_proposed.py

This will save the training patch 'clean_pats_rgb.npy' in the folder 'data/'

Then run the MFEBDN_Color model file using:

python MFEBDN_Color.py

This will save the 'MFEBDN_Color.h5' file of in the folder 'Pretrained_models/'.
