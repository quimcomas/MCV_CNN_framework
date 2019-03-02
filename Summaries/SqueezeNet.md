## Squeeze-and-excitation

[link to the code](https://github.com/hujie-frank/SENet)
[link to the paper](https://arxiv.org/pdf/1709.01507.pdf)

Squeeze and excitation networks define Convolutional Neural Networks with a novel architectural unit, the “Squeeze-and-Excitation” (SE) block, proposed in 2017 for the ImageNet Large Scale Visual Recognition Competition (paper from 2018) by Jie Hu, Li Shen, Samuel Albanie, Gang Sun and Enhua Wu. By implementing the SE unit, the team managed to win the ILSVRC 2017 by reducing the top-5 error to 2.251%.

The goal of SE blocks is to (more) efficiently exploit channel information, and increase network's sensitivity to informative features, to use them more effectively. They consist off 2 steps: Squeeze and Excitation.

#### Squeeze
The squeeze step proposes a way to embed global (spatial) information, by using global average pooling (to obtain channel-wise statistics). Its output can be interpreted as a set of local descriptors that share statistics with the whole image.
#### Excitation
The excitation step serves the purpose of adaptive recalibration. It aims to capture channel-wise dependencies, by making use of the output from the squeeze step. It is capable of learning the non linear interaction between channels, and, to ensure than multiple channels can be emphasised, it's also capable of learning their non-mutually-exclusive relationship. It maps the input-specific descriptor to a set of channel-specific weights, boosting the feature-discriminality.

### Summary

The SE blocks can be implemented in standard architectures, such as VGG etc. It allows for increasing the efficiency of the network, by only slightly increasing the computational cost.