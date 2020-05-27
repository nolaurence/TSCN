A tree based graph convolution network
=======================

## Get the original dataset:
### Retialrocket:
mkdir -p ~/.kaggle  
echo '{"username":"nolaurence","key":"581f65fa2f236667ac3989e5c525fbbb"}' > ~/.kaggle/kaggle.json  
chmod 600 ~/.kaggle/kaggle.json  

kaggle datasets download -d retailrocket/ecommerce-dataset
### Movielens 1m:
https://grouplens.org/datasets/movielens/1m/

## Required packages:
Python 3.7  
Pytorch 1.2 or later  
Pandas 0.24.2  
Numpy 1.16.2  

## How to run:
if you have downloaded raw data:  
1. specify the raw data directory in preprocess.py  
2. rm -r data
3. cd src
4. python preprocess.py
5. python main.py

if you want to run with existing data:
1.python main.py

