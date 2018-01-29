Convolutional Recurrent Neural Network for HTR
======================================

This software is clone implementation of the Convolutional Recurrent Neural Network (CRNN) in pytorch found [here](https://github.com/meijieru/crnn.pytorch)


Dependence
----------
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
* lmdb

Train a new model
-----------------
1. Construct dataset following origin guide. For training with variable length, please sort the image according to the text length.
2. ``python crnn_main.py [--param val]``. Explore ``crnn_main.py`` for details.
