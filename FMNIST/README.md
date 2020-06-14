# BPFC: Fashion MNIST

Run the following command to train a network with a modified LeNet architecture using the BPFC Regularizer:

`CUDA_VISIBLE_DEVICES=0 python2 bpfc_fmnist.py -EXP_NAME "bpfc_fmnist" -MAX_EPOCHS 50 -p_val 6 -l_ce 1 -l_reg 25 -mul_ru 1` 

## Trained Model Weights
BPFC trained model weights are available in bpfc_fmnist.pkl
