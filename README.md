# MetaAMI: A Novel Meta-Learning Approach for Predicting In-Hospital Mortality in Acute Myocardial Infarction

**MetaAMI** (A Novel Meta-Learning Approach for Predicting In-Hospital Mortality in Acute Myocardial Infarction)is an accurate and cost-effective meta-learning model for predicting in-hospital mortality in acute myocardial infarction patients using routinely collected clinical features. By combining random projection with stacked ensemble learning and an SVM meta-learner, **MetaAMI** provides reliable risk prediction and supports clinical decision-making for early mortality risk stratification.

## Flowchart of MetaAMI
![Flowchart of MetaAMI](Flowchart.png)

## Table of Contents
- [Installation](#installation)
- [Tutorials](#Tutorials)
- [Bug Report](#Bug-Report)
- [Authors](#Authors)
- [Publication](#Publication)
## Installation
1. Clone the MetaAMI git repository
```bash
git clone https://github.com/wan-mlab/MetaAMI.git
```
2. Navigate to the directory of MetaAMI package
```bash
cd /your path/MetaAMI
pip install .
```
## Tutorials
### Jupyter notebook
1. Modify the System Path and import module
```bash
import sys; sys.path.append('MetaAMI')
from MetaAMI import MetaAMI
```
2. unzip and read the test file
```bash
test = pd.read_csv('test.csv', index_col=0)
```
3. AMI risk prediction
```bash
MetaAMI.Predict(xxx)
```
   some explain.

4. Example Outputs

![Example Outputs](output.png)

The prediction results will be stored and exported to the Prediction_results.csv

### Predicting New Patient Samples
1. Prepare the input file
   Format: test result matrix with **test names** as columns and **patient id** as rows. Save your file as new_patient_testresult.csv.
2. Load your new patient data
```bash
import pandas as pd
new_patient = pd.read_csv('new_patient_testresult.csv', index_col=0)
```
3. Run prediction using the trained RanBALL model
```bash
from MetaAMI import MetaAMI
RanBALL.Predict(xxx)
```
   After running the command, the prediction results will appear in Prediction_results.csv
## Bug Report

If you find any bugs or problems, or you have any comments on MetaAMI, please don't hesitate to contact via email btuerhanbayi@unmc.edu or [Issues](https://github.com/wan-mlab/MetaAMI/issues).

## Authors
Bulidierxin Tuerhanbayi, Shibiao Wan

## Publication
xxx

## License 
xxxxx
