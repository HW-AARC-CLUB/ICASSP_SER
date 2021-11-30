![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)  

# A Novel End-to-End Speech Emotion Recognition Network with Stacked Transformer Layers

> Pytorch implementation for learning A Novel End-to-End Speech Emotion Recognition Network with Stacked Transformer Layers.


## Paper

[**A Novel End-to-End Speech Emotion Recognition Network with Stacked Transformer Layers**](links will update later...)

> contact with us: zeroroman711@gmail.com

## Usage  

### Prerequisites
~~~
pip install -r requirements.txt
~~~

### Datasets

You can get IEMOCAP datasets from [here](https://sail.usc.edu/iemocap/iemocap_release.htm).


### Before Training

1.Enter folder `5_fold_split`, and create empty folder `data` for IEMOCAP datasets.
~~~
mkdir data
~~~

And, put the downloaded datasets in 'data/'.

Organize the downloaded data files as per below structure:

data
|

├── Session1/
|

├── Session2/│
|

├── Session3/
|

├── Session4/
|

└── Session5/

2. Enter folder `data`, and create empty folder `features` to save extract audio features.

~~~
mkdir features
~~~

3.Run the file `5_fold_split.py` and `make_datasets.py` for audio pre-process feature.

Firstly, run
~~~
python 5_fold_split.py
~~~

and, then

~~~
python make_datasets.py
~~~
