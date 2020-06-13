# BPFC: MNIST

Run the following command to train a network with a modified LeNet architecture using the BPFC Regularizer:

`CUDA_VISIBLE_DEVICES=0 python2 bpfc_mnist.py -EXP_NAME "bpfc_mnist" -MAX_EPOCHS 50 -l_ce 1 -p_val 7 -l_reg 30 -mul_ru 1` 

## Trained Model Weights
BPFC trained model weights are available in bpfc_mnist.pkl
