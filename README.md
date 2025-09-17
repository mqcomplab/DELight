# DELight:  DEL-Imbalance Grouping of Heterogeneous Targets

This repository contains the code for the DELight method, which addresses the challenge of imbalanced heterogeneous targets in machine learning tasks. The method is designed to improve model performance by effectively grouping and handling imbalanced data.

First step is `gen_fingerprints.py` to generate fingerprints for the dataset. This script processes the input data and creates unique fingerprints that represent different groups within the dataset.

Second step is use `split.py` to split the dataset into training and testing sets. This script takes the original dataset and divides it into three parts, ensuring that each set is representative of the overall data distribution.

Third step is undersampling. If you are using cluster-based undersampling, then follow Workflow 1. Otherwise, if you are using isim undersampling, follow Workflow 2.

Undersampling Workflow 1: Cluster-based Undersampling
After splitting the dataset, use `cluster.py` to perform cluster-based undersampling. This script identifies clusters within the data. It starts at the highest threshold and iteratively lowers the threshold until the none of the actives are in singletons. 

Then, there are two options for the next step:
1. Max Sim: this will select clusters from highest threshold from previous clustering step and pick all the inactives from the clusters that contain actives. This is done in `max_sim.py`. It keeps selected until inactives is the same size as actives. If it is not enough, then it moves on to the next lower threshold and repeat the process until inactives is the same size as actives.

2. Min Sim: this will select clusters from the lowest threshold from previous clustering step and pick all the inactives from the clusters without contain actives. This is done in `min_sim.py`. It keeps selected until inactives is the same size as actives. If it is not enough, then it moves on to the next higher threshold and repeat the process until inactives is the same size as actives.

Undersampling Workflow 2: iSIM Undersampling

After splitting the dataset, use `under_isim.py` to perform iSIM undersampling. This script implements the iSIM algorithm for undersampling. It will gives two files as output: `neg_fps_isim_max.csv`, which are the inactive most similar to actives according to iSIM calculation, and `neg_fps_isim_min.csv`, which are the inactive least similar to actives according to iSIM calculation.

Finally, the model training is done in `train` folder. The `y_labels.py` script generates the labels for the training and testing sets based on the undersampled data. `rf.py`, `lr.py`, and `mlp.py` scripts are used to train Random Forest, Logistic Regression, and Multi-Layer Perceptron models, respectively. Each script takes the training data and labels as input and produces a trained model as output. We evaluate the models using accuracy, precision, recall, F1-score, and ROC-AUC metrics to see how well they perform on the test set using different undersampling methods.