# BPFC: CIFAR-10

Run the following command to train a ResNet-18 network using the BPFC Regularizer:

`CUDA_VISIBLE_DEVICES=0 python2 bpfc_cifar10.py -EXP_NAME "bpfc_cifar10" -MAX_EPOCHS 100 -p_val 5 -l_ce 1 -l_reg 1 -mul_ru 9` 

## Trained Model Weights
BPFC trained ResNet-18 model weights are available in bpfc_cifar10.pkl
