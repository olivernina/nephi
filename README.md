## Installation
1. Install conda (a local sandbox/install manager), create a conda enviroment

OS X:
```
$ brew install anaconda
$ echo ". /usr/local/anaconda3/etc/profile.d/conda.sh" >> ~/.bash_profile
```
Then
```
$  conda create --name nephi  python=2.7 
$  conda activate nephi
```

2. Install [PyTorch](http://pytorch.org/).
```
conda install pytorch torchvision opencv -c pytorch -y # this is enough if you don't need CUDA, else build from source
```
3. Install [lmdb](https://lmdb.readthedocs.io/en/release/), and a few more dependencies:

```
conda install -c conda-forge python-lmdb lxml python-levenshtein -y
```
4. Install WarpCTC as explained [here](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding).

```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build
cd build
cmake ../
make
cd ../pytorch_binding
python setup.py install
```
On OS X, substitute
```
cmake ../ -DWITH_OMP=OFF
```
remove -std=c++11 from setup.py file, and add
```
cd ../build
cp libwarpctc.dylib cp libwarpctc.dylib /usr/local/anaconda3/lib
```

Test it with
```
from warpctc_pytorch import CTCLoss
```
or this [gist](https://gist.github.com/rdp/bc27be54ec883109989426a9af79ca39).

This repository is a fork from the pytorch version of Convolutional Recurrent Neural Network (CRNN) repository found [here](https://github.com/meijieru/crnn.pytorch).
And from https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py (the original paper)

## Train a new model
1. Construct dataset following original guide. For training with variable length, please sort the image according to the text length.
2. To create the lmdb database, clone this repository and use ``create_dataset.py`` as follows: 

First fill a directory with training data, and one with validation data.  Example data structure:

```
/path/to/your/data/25_this is what it says.png
/path/to/your/data/26_this is what the next one says.jpg
```
Now bootstrap the lmdb (input) index databases:
```
nephi$  python create_dataset.py /path/to/your/training/data /new/train/lmdb/database
nephi$  python create_dataset.py /path/to/your/val/data /new/val/lmdb/database
```

If you'd like to input from XML descriptions (ex: XML that describes line portions within a larger image), 
add --xml at the end

```
python create_dataset.py /path/to/your/val/data /new/val/lmdb/database --xml
```

3. To train a new model, we execute `crnn_main.py`. The argument format is as follows:
```
nephi$ python crnn_main.py --trainroot /new/train/lmdb/database --valroot /new/val/imdb/database [--cuda]
```

It will train using your trainroot data, backpropagating to the neural network every "batch size" images (default 64), and update with how well it's doing as it goes.

The `--cuda` flag enables GPU acceleration. If your machine has CUDA and you do not use this flag, the software will warn you that you could be using GPU acceleration.

Be sure to **provide a valid alphabet.txt file** for your dataset (either pass one in as a parameter or create local file alphabet.txt). 

For more help with argument structure, use `nephi$ python crnn_main.py -h`.
