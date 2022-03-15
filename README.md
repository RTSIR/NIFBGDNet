# NIFBGDNet
This repo contains the KERAS implementation of "Multi Scale Pixel Attention and Feature Extraction based Neural Network for Image Denoising"


# Run Experiments

To test for blind gray denoising using MFEBDN write:

python Test_gray.py

The resultant images will be stored in 'Test_Results/Gray/'

To test for blind color denoising using MFEBDN write:

python Test_color.py

The resultant images will be stored in 'Test_Results/Color/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database is also displayed in the console as output.

# NIFBGDNet architecture image
![MFEBDN_Architecture](https://user-images.githubusercontent.com/89151608/148807575-b518568c-0079-4e6b-ac8b-29c01972e40e.png)

# NIFBGDNet gray image denoising comparison
![MFEBDN_Gray_Denoising_Comparison](https://user-images.githubusercontent.com/89151608/148807795-c09c5b4c-5476-4f08-b27e-2c40d95bf035.png)

# NIFBGDNet color image denoising comparison
![MFEBDN_Color_Denoising_Comparison](https://user-images.githubusercontent.com/89151608/148807881-faf501ee-4699-4978-a2df-0ef046e9c1a6.png)

# NIFBGDNet ablation study
![MFEBDN_Ablation_Study](https://user-images.githubusercontent.com/89151608/148814752-6027041e-cc6a-411e-a1b2-5aed5fb33508.png)

# NIFBGDNet model size comparison with other blind denoising networks
![MFEBDN_Model_Size_Comparison](https://user-images.githubusercontent.com/89151608/148814995-1d49c393-9039-4a85-961d-4a722f496600.png)
