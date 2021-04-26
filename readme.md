# Comprehensive analysis of CrypTen for Neural Network Inference
With the advert of Deep Learning, more and more privacy concerns are raised. This leads to many proposed solutions. CrypTen is one of the latest proposed solution/library, which doesn't have proper evaluation on it's performance. Therefore, we attempt to analyze CrypTen for NN inference task.

This project is a part of the **"CSE 598: Secure Computation of Machine Learning"** course at ASU.

# Installation

Please refer following instructions for installation of CrypTen and running the scripts.
For CrypTen installation:

    $ git clone https://github.com/facebookresearch/CrypTen.git
    $ cd ./CrypTen/
    $ git checkout 0cd3b3defb63dace4b8b54bef8158aa9a5924eb7
    $ pip3 install .

To run the scripts in copy this project folder to CrypTen home directory.

To run any python scripts given in `CrypTen_v2` folder copy and past them to the CrypTen home directory. Moreover, to avoid failures on GPU machine, run the scripts with following command:

    CUDA_VISIBLE_DEVICES=-1 python3 <script_name>.py

 

## Dataset
We pre-train our models on [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).

## Python scripts
Please refer to the following folder structure for running different scripts:
1. `training` : This folder contains training scripts for ResNet32 and MiniONN on COVID x-ray dataset.
2. `CrypTen_v1` : This folder contains the various evaluation notebooks for NN inference using CrypTen with implementation version 1.
3. `CrypTen_v2` : This folder contains the more secure evaluation scripts which can be used with different number of servers and better re-presents the security instance then v1.

