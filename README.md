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

# MFEBDN architecture image
![MFEBDN_Architecture](https://user-images.githubusercontent.com/89151608/148807575-b518568c-0079-4e6b-ac8b-29c01972e40e.png)

# MFEBDN gray image denoising comparison
![MFEBDN_Gray_Denoising_Comparison](https://user-images.githubusercontent.com/89151608/148807795-c09c5b4c-5476-4f08-b27e-2c40d95bf035.png)

# MFEBDN color image denoising comparison
![MFEBDN_Color_Denoising_Comparison](https://user-images.githubusercontent.com/89151608/148807881-faf501ee-4699-4978-a2df-0ef046e9c1a6.png)

# MFEBDN ablation study
![MFEBDN_Ablation_Study](https://user-images.githubusercontent.com/89151608/148814752-6027041e-cc6a-411e-a1b2-5aed5fb33508.png)

# MFEBDN model size comparison with other blind denoising networks
![MFEBDN_Model_Size_Comparison](https://user-images.githubusercontent.com/89151608/148814995-1d49c393-9039-4a85-961d-4a722f496600.png)
