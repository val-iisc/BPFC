# Bit Plane Feature Consistency Regularizer 
This repository contains code for the implementation of our paper titled "Towards Achieving Adversarial Robustness by Enforcing Feature Consistency Across Bit Planes", accepted at CVPR 2020. Our paper is available on arXiv [here](https://arxiv.org/abs/2004.00306).

Our proposed Bit Plane Feature Consistency (BPFC) regularizer achieves adversarial robustness without the use of adversarial samples during training. Prior works which attempt to achieve this have been broken by adaptive attacks subsequently.

<p align="center">
    <img src="https://github.com/GaurangSriramanan/BPFC/blob/master/BPFC_schematic_figure.png" width="800"\>
</p>

# Robustness of BPFC Trained Models
Clean and PGD 1000-step accuracy of BPFC trained models and PGD Adversarially trained models across different datasets:


<table>
<thead>
  <tr>
    <th></th>
    <th colspan="2">CIFAR-10</th>
    <th colspan="2">F-MNIST</th>
    <th colspan="2">MNIST</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Method</td>
    <td>Clean</td>
    <td>PGD-1000</td>
    <td>Clean</td>
    <td>PGD-1000</td>
    <td>Clean</td>
    <td>PGD-1000</td>
  </tr>
  <tr>
    <td>BPFC</td>
    <td>82.4%</td>
    <td>34.4%</td>
    <td>87.2%</td>
    <td>67.7%</td>
    <td>99.1%</td>
    <td>85.7%</td>
  </tr>
  <tr>
    <td>PGD</td>
    <td>82.7%</td>
    <td>47.0%</td>
    <td>87.5%</td>
    <td>79.1%</td>
    <td>99.3%</td>
    <td>   94.1%<br></td>
  </tr>
</tbody>
</table>


# Environment Settings
+ Python: 2.7.15
+ PyTorch: 1.0.0
+ TorchVision: 0.2.2.post3
+ Numpy: 1.17.2
