Installation
======================================

1. Create a conda enviroment

``conda create --name nephi``

``source activate nephi``

2. Install Pytorch

3. Install lmdb

4. Install WarpCTC as follows (Taken from https://github.com/pytorch/pytorch#installation).

`WARP_CTC_PATH` should be set to the location of a built WarpCTC
(i.e. `libwarpctc.so`).  This defaults to `../build`, so from within a
new warp-ctc clone you could build WarpCTC like this:

```bash
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
```

Otherwise, set `WARP_CTC_PATH` to wherever you have `libwarpctc.so`
installed. If you have a GPU, you should also make sure that
`CUDA_HOME` is set to the home cuda directory (i.e. where
`include/cuda.h` and `lib/libcudart.so` live). For example:

```
export CUDA_HOME="/usr/local/cuda"
```

Now install the bindings:
```
cd pytorch_binding
python setup.py install
```

Example to use the bindings below.

```python
    from warpctc_pytorch import CTCLoss
```

This software is clone implementation of the Convolutional Recurrent Neural Network (CRNN) in pytorch found [here](https://github.com/meijieru/crnn.pytorch)



Dependencies
------------
* lmdb
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)


Train a new model
-----------------
1. Construct dataset following origin guide. For training with variable length, please sort the image according to the text length.
2. ``python crnn_main.py [--param val]``. Explore ``crnn_main.py`` for details.
