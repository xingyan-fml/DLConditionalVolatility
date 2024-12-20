This repository contains the code used for the paper "Deep Learning of Conditional Volatility and Negative Risk-Return Relation" by Wenxuan Ma, Qi Wu, and Xing Yan (2024). Please cite this paper if you are using the code:

Wenxuan Ma, Qi Wu, and Xing Yan. Deep Learning of Conditional Volatility and Negative Risk-Return Relation (September 14, 2024). Preprint.


Data Source and Preparation

The data used in our paper is from Jensen et al. (2023) (Is There a Replication Crisis in Finance?). To get the data, use the code provided by the authors at https://github.com/bkelly-lab/ReplicationCrisis.

Follow the instructions in "How to Generate Global Stock Returns and Stock Characteristics" at https://github.com/bkelly-lab/ReplicationCrisis/tree/master/GlobalFactors. Focus on the U.S. market data only. Run the .sas code to get the usa.csv file and place it in our "rawData" folder.


Steps for Replicating Results

1. Setup: Make sure you know how to use Python and have installed the following packages: numpy, pandas, PyTorch, statsmodels, etc.

2. Data Preprocessing: Run preprocessing.py located in the "preprocessing" folder. This will create an usa_new.csv file with data of 153 firm characteristics and 1-month-ahead excess returns, as used in Jensen et al. (2023).

3. Neural Network Training and Prediction: Run exper_hnn.py in the "hnn" folder. This script performs neural network training and prediction. A GPU is recommended, preferably an RTX 3090 or 4090. The process may take one or two days. The "results" folder will be created with the necessary outcomes.

4. Result Collection: Run the .py scripts in each "collect*" folder. These scripts will generate all the numbers, tables, and figures in our paper.


Note: There may be some randomness in the results, but overall, they should be quite close to those reported in the paper.
