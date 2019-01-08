# Gleason
---
![plot](https://user-images.githubusercontent.com/16754088/50859189-ad7dc600-1360-11e9-9ee6-bbc37649d107.png)
---
- create\_crops\_normal.py - creates normal crops from patholog svs slide files
- create\_crops\_tumor.py	- creates malginant crops from patholog svs slide files
- create\_prostate\_normal.sh	- bash script with normal slides to process
- create\_prostate\_tumor.sh	- bash script with malignant slides to process
- prepare_tfrecords.py - script to encode image data into tfrecord format
- resnet\_config.py	- script that configures hyperparameters and augmentations
- resnet\_utils.py - util functions for ResNet models
- resnet\_v2.py - implementation of ResNet model
- test\_resnet.py	- script to test trained ResNet model performance
- tfrecord.py	- script for data augmentations and encoding and decoding tfrecords
- train\_class\_weights.npy	- weights for gleason classes
- train\_resnet.py - script to train a ResNet model 
- unet\_preprocess.py - implementation of Unet used for normalizing images
---
