# CIL-Project
ETHz CIL 2023 Collaborative Filtering

## 1. Installation: Setup the environment
### Create environment and install dependencies
```
conda create --name cil python=3.9
conda activate cil
pip install -r requirements.txt
```
### Download dataset
Create a directory "/data". Then put "data_train.csv" and "sampleSubmission.csv" inside. 

## 2. Train a model and generate predictions
```
python train.py config.yaml
```
