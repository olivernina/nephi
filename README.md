Installation
======================================

1. Create a conda enviroment
``conda create --name nephi``
``source activate nephi``

2. Install Pytorch

3. Install lmdb

4. Install WarpCTC as explained [here](Taken from https://github.com/pytorch/pytorch#installation).
git clone https://github.com/SeanNaren/warp-ctc.git


This software is clone implementation of the Convolutional Recurrent Neural Network (CRNN) in pytorch found [here](https://github.com/meijieru/crnn.pytorch)



Dependencies
------------
* lmdb
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)


Train a new model
-----------------
1. Construct dataset following origin guide. For training with variable length, please sort the image according to the text length.
2. ``python crnn_main.py [--param val]``. Explore ``crnn_main.py`` for details.
