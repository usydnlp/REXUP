# REXUP: I REason, I EXtract, I UPdate with Structured Compositional Reasoning for Visual Question Answering

<p align="center">
  <b>Siwen Luo, Soyeon Caren Han, Kaiyuan Sun, Josiah Poon</b></span>
</p>

**_For any issue related to code, pls first search for solution in Issues section, if there is no result, pls post a comment in "Issues" section, we will help soon._**

This is an implementation of the [REXUP network](https://arxiv.org/abs/2007.13262) to take advantage of scene graph from the <b>[the GQA dataset](https://www.visualreasoning.net)</b>. GQA is a new dataset for real-world visual reasoning, offrering 20M diverse multi-step questions, all come along with short programs that represent their semantics, and visual pointers from words to the corresponding image regions. Here we extend the MAC network to work over VQA and GQA, and provide multiple baselines as well.

**_decription for REXUP NETWORK_**
The REXUP network contains two parallel branches, object-oriented branch and scene-graph oriented branch. Each branch contains a sequence of REXUP cells where each cell operates for one reasoning step for the answer prediction. 

**_replace the image here_**

<div align="center">
  <img src="https://raw.githubusercontent.com/usydnlp/REXUP/master/model_img/REXUP.png" style="float:left" width="260px" height='360px'>
  <img src="https://raw.githubusercontent.com/usydnlp/REXUP/master/model_img/REXUP_cell.png" style="float:right" width="500px", height='280px'>
</div>

## Bibtex

For the REXUP Paper:

```
@article{siwen2020rexup,
  title={REXUP: I REason, I EXtract, I UPdate with Structured Compositional Reasoning for Visual Question Answering},
  author={Siwen Luo, Soyeon Caren Han, Kaiyuan Sun and Josiah Poon},
  conference={International Conference on Neural Information Processing},
  year={2020}
}
```

## Requirements

**Note: To make sure that you can reimplement our model and result, we recommand you to use Dockerfile and Makefile we provide in this github to create a same environment as we conduct.**

- We have performed experiments on Titan RTX GPU with 24GB of GPU memory, from experiments, our model needs around 20GB GPU memory with 128 batch size, you can reduce the batch size to reduce the total memory you need.

Let's begin from cloning this reponsitory branch:

```
git clone XX
```

- run `pip install docker` to install docker in your server.
- cd into the folder, run `sudo build make` to create an image of the location.
- use `sudo make` to start a docker image. (if you have any issue related to GPU-support dokcer image, pls refer to [docker GPU website](https://www.tensorflow.org/install/docker))


## Pre-processing

Before training the model, you have to download the GQA dataset and extracted features for the images:

### Dataset

To download and unpack the data, run the following commands:

<!-- ```bash
mkdir data
cd data
wget https://nlp.stanford.edu/data/gqa/data1.2.zip
unzip data1.2.zip
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ../
```

#### Notes
1. **The data zip file here contains only the minimum information and splits needed to run the model in this repository. To access the full version of the dataset with more information about the questions as well as the test/challenge splits please download the questions from the [`official download page`](https://www.visualreasoning.net/download.html).**


```bash
mkdir data
cd data
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove
mkdir gqa
cd gqa
wget https://nlp.stanford.edu/data/gqa/sceneGraphs.zip
unzip sceneGraphs.zip -d sceneGraphs
wget https://nlp.stanford.edu/data/gqa/questions1.3.zip
unzip questions1.3.zip -d questions
```

Alternatively, if you have the latest version of the GQA dataset already downloaded, use symlinks to link to the dataset items.

### Feature download and preparation

<!-- 
```bash
cd data
wget http://nlp.stanford.edu/data/gqa/objectFeatures.zip
unzip objectFeatures.zip
cd ../
python merge.py --name objects
``` -->

## Training

To train the model, run the following command:

```bash
python main.py --expName "gqaExperiment" --train --testedNum 10000 --epochs 25 --netLength 4 @configs/gqa/gqa_ensemble.txt
```

First, the program preprocesses the GQA questions. It tokenizes them and maps them to integers to prepare them for the network. It then stores a JSON with that information about them as well as word-to-integer dictionaries in the `data` directory.

Then, the program trains the model. Weights are saved by default to `./weights/{expName}` and statistics about the training are collected in `./results/{expName}`, where `expName` is the name we choose to give to the current experiment.

### Notes

- The number of examples used for training and evaluation can be set by `--trainedNum` and `--testedNum` respectively.
- You can use the `-r` flag to restore and continue training a previously pre-trained model.
- We recommend you to try out varying the number of REXUP cells used in the network through the `--netLength` option to explore different lengths of reasoning processes.
- Good lengths for GQA are in the range of 2-6.

See [`config.py`](config.py) for further available options (Note that some of them are still in an experimental stage).


## Evaluation

To evaluate the trained model, and get predictions and attention maps, run the following:

```bash
python main.py --expName "gqaExperiment" --finalTest --testedNum 10000 --netLength 4 -r --getPreds --getAtt @configs/gqa/gqa_ensemble.txt
```

The command will restore the model we have trained, and evaluate it on the validation set. JSON files with predictions and the attention distributions resulted by running the model are saved by default to `./preds/{expName}`.

- In case you are interested in getting attention maps (`--getAtt`), and to avoid having large prediction files, we advise you to limit the number of examples evaluated to 5,000-20,000.


