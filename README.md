# nlp_d2l_practice

This is a repository for nlp's practice of the book [Dive into Deep Learning](https://d2l.ai/).

## Installation
```bash
conda create --name d2l python=3.9 -y
conda install numpy scipy scikit-learn matplotlib tqdm opencv pandas
conda install tensorboard jupyter seaborn
# MPS acceleration is available on MacOS 12.3+
conda install pytorch torchvision -c pytorch
pip install d2l==1.0.0b0
```

## Pretraining
### Word2Vec
- Skip-gram
- CBOW

### Approximate Training
- Negative Sampling
- Hierarchical Softmax
