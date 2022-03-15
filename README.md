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

# NIFBGDNet architecture image
![image](https://user-images.githubusercontent.com/89151608/158437632-58c4ee5c-153f-4799-b16d-e60019cbd028.png)

![image](https://user-images.githubusercontent.com/89151608/158442819-86c2185b-963d-480e-aace-c6e7ba9fb5d4.png)


# NIFBGDNet gray image denoising comparison
![image](https://user-images.githubusercontent.com/89151608/158437994-34fd5924-4ba8-4aad-b0d8-561a9217b119.png)

# NIFBGDNet color image denoising comparison
![image](https://user-images.githubusercontent.com/89151608/158438118-973312e0-70f0-4113-b74b-f15906a7a117.png)
