# US-famli
Ultra sound Fetal Age Machine Learning Initiative

## How to train a network from scratch

1. Convert RGB to lum (US-famli/src/py/rgb2lum.py)
2. Create CSV file with columns image,image1 (image -> Input to U-net, image1-> Output)
3. Convert to tfRecords from single scalar file US-famli/src/py/dl/tfRecords.py
4. Train a network with Unet architecture US-famli/src/py/dl/train.py

