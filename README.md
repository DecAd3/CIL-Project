# CIL-Project
ETHz CIL 2023 Collaborative Filtering

## 1. Environment setup
### Create environment and install dependencies
```
conda create --name cil python=3.9
conda activate cil
pip install -r requirements.txt
```
### Download dataset 
Create a directory `/data`. Then put `data_train.csv` and `sampleSubmission.csv` inside. 

## 2. Kaggle result reproduction
TODO: Which files to run/ How to run? Steps. 

## 3. Details implementation
### 3.1 Train a single model
#### Check the settings in `config.yaml`:  
`experiment_args/model_name`: Model name.  
`experiment_args/generate_submissions`: True: Use the entire dataset to generate submissions; False: Split the dataset for validation. 
#### Run the following command: 
```
python train.py config.yaml
```
### 3.2 Apply grid search to a single model
#### Check the settings in `config_cv.yaml`:  
`experiment_args/model_name`: Model name.  
TODO:
#### Run the following command: 
```
python grid_search.py config_cv.yaml
```

### 3.3 Ensemble
#### 3.3.1 Save cross validation results
#### Check the settings in `config_cv.yaml`:  
`experiment_args/model_name`: Model name.  
`experiment_args/model_instance_name`: A prefix for saving prediction filenames.  
`experiment_args/save_full_pred`: True: The prediction values of fold x will be saved in path `ensemble_args/data_ensemble + experiment_args/model_instance_name + "_fold_{fold number}_train/test".txt`. The train/test in the file name means the prediction results of ids provided in the `data_train.csv` and `sampleSubmission` respectively. 
#### Run the following command for cross validation: 
```
python cross_validation.py config_cv.yaml
```

[comment]: <> (Check settings in the `config_cv.yaml` file. Modify attribute `experiment_args/model_name` value and set it to the model name you want to run for cross validation. Modify attribute `experiment_args/model_instance_name` value and it will be used as a prefix for saved prediction filenames. If `experiment_args/save_full_pred` is set to `True`, the prediction values of fold x will be saved in path `ensemble_args/data_ensemble + experiment_args/model_instance_name + "_fold_{fold number}_train/test".txt`. The train/test in the file name means the prediction results of ids provided in the `data_train.csv` and `sampleSubmission` respectively. To run cross validation, using the code)

#### 3.3.2 Perform ensemble
#### Check the settings in `config.yaml`:  
`experiment_args/model_name`: ensemble.  
TODO:
#### Run the following command for ensemble:
```
python train.py config.yaml
```

