# Bit Plane Feature Consistency Regularizer 
A repository that implements adversary-free training of robust Neural Networks using the Bit Plane Feature Consistency (BPFC) Regularizer, proposed in our paper "Towards Achieving Adversarial Robustness by Enforcing Feature Consistency Across Bit Planes", accepted at CVPR-2020. Our paper is available on arXiv [here](https://arxiv.org/abs/2004.00306).

![schematic](https://github.com/GaurangSriramanan/BPFC/blob/master/BPFC_schematic_figure.png)

# Robustness of BPFC Trained Models
Clean and Robust accuracy of BPFC trained models and PGD Adversarially trained models across different datasets:

|: Method :|:     CIFAR 10    :|:     F MNIST     :|:      MNIST      :| 
|   :---:  |:---:   |:---:     | :---:  |:---:     |  :---: |:---:     |
|          |  Clean | PGD 1000 |  Clean | PGD 1000 |  Clean | PGD 1000 |
|   BPFC   |  82.4% |   34.4%  |  87.2% |   67.7%  |  99.1% |   85.7%  | 
|   PGD    |  82.7% |   47.0%  |  87.5% |   79.1%  |  99.3% |   94.1%  |

# Environment Settings
+ Cuda: 10.1
+ Python: 2.7.15
+ PyTorch: 1.0.0
+ TorchVision: 0.2.2.post3
+ Numpy: 1.17.2
