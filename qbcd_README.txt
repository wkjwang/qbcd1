# Copyright (c) 2023  Kaijun WANG  [wkjwang@qq.com]
# All rights reserved.  See the file qbcd_Copyright.txt for license terms. 

This python code package is the software of the QBCD method:
“Causal Direction inference by the Prediction Bias of Quantile Regression”
about inferring the causal direction between two variables ( X→Y or Y→X)
from a dataset corresponding to two variables (X and Y)

## Python Dependencies
- numpy
- pandas
- scikit-learn (sklearn)
- scipy.stats

## Data
Data benchmarks about cause-effect pairs are available from:
- simulated SIM-group & AN-group benchmarks
  https://github.com/tagas/bQCD
- Tubingen Database with cause-effect pairs
  https://webdav.tuebingen.mpg.de/cause-effect/


## The structure of this package:
qbcd_load_data.py 	--- load the data sets mentioned above
qbcd_bias_infer.py 	--- the code about causal direction inference by QBCD
qbcd_utils.py 	--- the functions used by QBCD
qbcd_main.py	--- the experiments from the related paper, please download Data benchmarks
qbcd_example.py 	--- example of QBCD to infer the causal direction of one dataset

solution		--- folder for the result files of code-running
data		--- folder for the data sets

## Citation
If this package is useful in your work, please consider the citing.
Kaijun WANG, et al. Causal Direction inference by the Prediction Bias of Quantile Regression. 2024
