# PISNet
We proposed a prior-induced interference suppression network (PISNet) to achieve RFI suppression and useful signal recovery in time-frequency domain. A dataset called SarTF is built to address the difficulty of data shortage to support the training of PISNet.

## SarTF Dataset
A dataset of the time-frequency spectrograms of SAR echoes with RFI, which can be used to train the end-to-end deep neural networks. 

* Download <br>
BaiduNetdisk: https://pan.baidu.com/s/1cONUOWE60QoHzs-R_tdc5w  Password: foyr

## Code
* Currently the network is suitable for dealing with interference intensity similar to that in SarTF Dataset. If the difference in the ISR (Signal-Interference-Ratio) is too large (eg. the SIR too low), the performance may be degraded. The SIR of the training set should be adjusted accordingly. 
* For specific training details, please refer to the experiment and discussion chapters of our paper.

## Cite
If you want to use this SarTF dataset or use PISNet as contrast model, please cite as follows

> J. Shen, B. Han, Z. Pan, G. Li, Y. Hu and C. Ding, "Learning Time–Frequency Information With Prior for SAR Radio Frequency Interference Suppression," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022, Art no. 5239716.
