
# DUVRec

This repository is the implementation of our paper's model 'Contrastive Learning of Dual-view User Representation for Sequential Recommendation'.

### Files in the folder

- `data/`
  - `steam/`
    - `steam_reviews.json.gz`: raw rating file and metadata of Steam dataset;
    - `preprocess.py`: the data preprocess script;
    - `gen_test_neg.py`
    - `attr_file.py`
  - `movielens/`
    - `ratings.dat`: raw rating file of Movielens dataset(UserID::MovieID::Rating::Timestamp);
    - `movies.dat`: genre file of Movielens dataset(MovieID::Title::Genres);
    - `preprocess.py`: the data preprocess script;
    - `gen_test_neg.py`
    - `attr_file.py`
  - `toys/`
    - `Toys_and_Games.json.gz`: raw review file of Toys dataset;
    - `meta_Toys_and_Games.json.gz`: raw metadata of Toys dataset;
    - `preprocess.py`: the data preprocess script;
    - `gen_test_neg.py`
    - `attr_file.py`
- `model/`
  - `main.py`
  - `config.py`
  - `...`
- `weights/`


### Required packages
The code has been tested running under Python 3.6.7, with the following packages installed (along with their dependencies):
- torch == 1.7.1+cu101
- numpy == 1.18.5

### Running Procedure

#### Prepare data
Steam, Movielens, Amazon Toys&Games dataset can be respectively downloaded from 
'http://cseweb.ucsd.edu/jmcauley/datasets.html#steam_data/'
'https://grouplens.org/datasets/movielens/'
'http://deepyeti.ucsd.edu/jianmo/amazon/index.html'

Please first put data files of Steam, Movielens and Toys into `data/steam`, `data/movielens` ans `data/toys`, then run `preprocess.py`, `gen_test_neg.py`, `attr_file.py` successively to process data preparation. 最终`data/steam`,`data/movielens`和`data/toys`中的数据文件应包含`attr_test.txt`, `attr_train.txt`, `test_candidate_1_50.npy`, `test.txt`, `train.txt`。

for convinence, we also upload the preprocessed data

#### Run DUVRec
```
$ cd src
$ python model/main.py
```
The settings of datasets and parameters can be altered in `model/config.py`. 
The model checkpoints and training log will be saved in `weight/`.

