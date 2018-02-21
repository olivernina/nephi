## Installation
1. Create a conda enviroment

install conda:
OS X:
```
$ brew install anaconda
$ echo ". /usr/local/anaconda3/etc/profile.d/conda.sh" >> ~/.bash_profile
```
Then
```
$  install conda:
$  conda create --name nephi
$  source activate nephi
```

2. Install [PyTorch](http://pytorch.org/).

3. Install [lmdb](https://lmdb.readthedocs.io/en/release/).

4. Install WarpCTC as explained [here](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding).
Make sure to update the link to the repository in the instructions to https://github.com/SeanNaren/warp-ctc.git 

This repository is a fork from the Convolutional Recurrent Neural Network (CRNN) repository found [here](https://github.com/meijieru/crnn.pytorch)



## Dependencies
* lmdb
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
* lxml


## Train a new model
1. Construct dataset following original guide. For training with variable length, please sort the image according to the text length.
2. To create the lmdb database, clone this repository and use ``create_dataset.py`` as follows:  
```
nephi$  python create_dataset.py /path/to/your/training/data /new/train/lmdb/database
nephi$  python create_dataset.py /path/to/your/val/data /new/val/lmdb/database
```
3. To train a new model, we execute `crnn_main.py`. The argument format is as follows:
```
nephi$ python crnn_main.py --trainroot /new/train/lmdb/database --valroot /new/val/imdb/database [--cuda]
```
The `--cuda` flag enables GPU acceleration. If your machine has CUDA and you do not use this flag, the software will warn you that you could be using GPU acceleration. Be sure to **provide a valid alphabet.txt file** for your dataset. For more help with argument structure, use `nephi$ python crnn_main.py -h`.
