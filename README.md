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
#### Run the following commands: 
```
python cross_validation.py config_cv.yaml
python train.py config.yaml
```
The result will be in directory `/output/submission`.  

## 3. Details implementation
### 3.1 Train a single model
#### Check the settings in `config.yaml`:  
`experiment_args/model_name`: Model name.  
`experiment_args/generate_submissions`: True: Use the entire dataset to generate submissions; False: Split the dataset for validation. 
#### Run the following command: 
```
python train.py config.yaml
```
Config file for NCF: `ncf.yaml` 
Please note that when training an NCF model, an unfortunate crash might occur randomly due to either a zero-shaped tensor or segmentation fault. If that happens, please simply rerun the training. 

### 3.2 Apply cross validation to a single model
#### Check the settings in `config_cv.yaml`:  
`experiment_args/model_name`: Model name.  
`experiment_args/generate_submissions`: False.  
`experiment_args/save_full_pred`: False.  
`ensemble_args/fold_number`: Fold number

#### Run the following command for cross validation: 
```
python cross_validation.py config_cv.yaml
```

### 3.3 Apply grid search to a single model
#### Check the settings in `config_cv.yaml`:  
`experiment_args/model_name`: Model name.  
`experiment_args/generate_submissions`: False.  
`experiment_args/save_full_pred`: False.  
`ensemble_args/fold_number`: Fold number

#### Check the settings in `grid_search.py`:  
Modify the parameters in function `grid_search`.  
#### Run the following command for grid search: 
```
python grid_search.py config_cv.yaml
```

### 3.4 Ensemble
#### 3.4.1 Save cross validation results
#### Check the settings in `config_cv.yaml`:  
`experiment_args/model_name`: Model name.  
`experiment_args/model_instance_name`: A prefix for saving prediction filenames.  
`experiment_args/save_full_pred`: True: The prediction values of fold x will be saved in path `ensemble_args/data_ensemble + experiment_args/model_instance_name + "_fold_{fold number}_train/test".txt`. The train/test in the file name means the prediction results of ids provided in the `data_train.csv` and `sampleSubmission` respectively. 
#### Run the following command for cross validation: 
```
python cross_validation.py config_cv.yaml
```

[comment]: <> (Check settings in the `config_cv.yaml` file. Modify attribute `experiment_args/model_name` value and set it to the model name you want to run for cross validation. Modify attribute `experiment_args/model_instance_name` value and it will be used as a prefix for saved prediction filenames. If `experiment_args/save_full_pred` is set to `True`, the prediction values of fold x will be saved in path `ensemble_args/data_ensemble + experiment_args/model_instance_name + "_fold_{fold number}_train/test".txt`. The train/test in the file name means the prediction results of ids provided in the `data_train.csv` and `sampleSubmission` respectively. To run cross validation, using the code)

#### 3.4.2 Perform ensemble
#### Check the settings in `config.yaml`:  
`experiment_args/model_name`: ensemble.  
TODO:
#### Run the following command for ensemble:
```
python train.py config.yaml
```

