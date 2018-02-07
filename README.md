Installation
======================================

1. Create a conda enviroment
``conda create --name nephi``
``source activate nephi``

2. Install Pytorch

3. Install lmdb

4. Install WarpCTC as explained [here](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding).
Make sure to update the link to the repository in the instructions to https://github.com/SeanNaren/warp-ctc.git 

This repository is a fork from the Convolutional Recurrent Neural Network (CRNN) repository found [here](https://github.com/meijieru/crnn.pytorch)



Dependencies
------------
* lmdb
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)


Train a new model
-----------------
1. Construct dataset following original guide. For training with variable length, please sort the image according to the text length.
2. ``python crnn_main.py [--param val]``. Explore ``crnn_main.py`` for details.
