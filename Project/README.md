# fake-news-attack

Model of fake news dectetion and attack it with [textattack](https://github.com/QData/TextAttack).  
Debug on win11 with cuda-11.2, tensenflow-2.10.1 and you can download all dependence on requirements.txt  

### Dataset
We use [kaggle-fake-news](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and [Liar](https://arxiv.org/abs/1705.00648v1),   
please download [kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/download?datasetVersionNumber=1) or [Liar](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) and unzip them into dataset path.  
We also upload the data on [dirve](https://drive.google.com/drive/folders/1T58dHhgk6IDoOZAH4lMSgvb42V1eRlbA?usp=sharing)  

### Word Embedding, Network, TextAttack
Detail in [notebook](https://github.com/H3CO3/fake-news-attack/blob/main/notebook.ipynb)  
If you run in colab:  
```
! pip install tensorflow_text
! pip install textattack
```

### Attack
Try [fake-news-explainability](https://github.com/ljyflores/fake-news-adversarial-benchmark) attack  
Detail in [notebook](https://github.com/H3CO3/fake-news-attack/blob/main/RawNegProcess.ipynb) and [notebook](https://github.com/H3CO3/fake-news-attack/blob/main/Project.ipynb)

### Group Members
MQY, ZSY, LYW
