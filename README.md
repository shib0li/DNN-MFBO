# *DNN-MFBO*: Multi-Fidelity Bayesian Optimization via Deep Neural Networks

by Shibo Li, Wei Xing, Mike Kierby and Shandian Zhe

This is the python implementation of the our papr [Multi-Fidelity Bayesian Optimization via Deep Neural Networks](https://arxiv.org/abs/2007.03117). Bayesian Optimization(BO) is a popular framework to optimize black-box functions. In many applications, the objective function can be evaluated at multiple fidelities to enable a trade-off between the cost and accuracy. We preposed DNN-MFBO that can flexibly capture all kinds of complicated relationships between the fidelities to improve the objective function estimation and hence the optimization performance, please refer our paper for more details.

## System Requirement
We tested our code with python 3.6 on Ubuntu 18.04. Our implementation relies on TensorFlow 1.15. Other packages include scikit-learn for data standarlization and hdf5stroage for saving the results to mat file. Please use pip or conda to install those dependencies. 

```
pip install hdf5storage
```
```
pip install scikit-learn
```
We highly recommend to use [_Docker_](https://www.docker.com/) to freeze the running experiments. We attach our docker build file.

## Run
Please find the details of running configuration from *run-*.sh* 

## License
DNN-MFBO is released under the MIT License, please refer the LICENSE for details

## Contact
If you have any questions, please email me at shibo 'at' cs.utah.edu, or create an issue on github. The datasets used of the last two applications in the paper are proprietary datasests, please contact our data provider if you are interested. We attach examples of three well-known sythetic multi-fidelity functions from https://www.sfu.ca/~ssurjano/index.html




