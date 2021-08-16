# LOSA-DNN

A lightweight one-in-two stream attention-based deep neural network (LOSA-DNN) is proposed for the real-time human action recognition. The proposed LOSA-DNN is composed of 1) RGB image input without optical flow image, 2) EfficientNet B3, and 3) attention-based two-stream network consisting of 2D separable convolution neural network (CNN) and long-short term memory (LSTM) with two-path pooling. 

<p align="center">
<img src="https://github.com/farhantandia/LOSA-DNN/blob/main/network.jpg"><br>
</p>

## Dependencies
- Python 3.7
- Tensorflow 2.3
- tensorflow-addons
- [efficientnet](https://github.com/qubvel/efficientnet)
- [keras-video-generators](https://github.com/metal3d/keras-video-generators)
- [keras_self_attention](https://github.com/CyberZHG/keras-self-attention)

## Usage
Download dataset HMDB-51 and UCF-101 and split the training and testing data inside each folder. After that You can simply run the code:
<pre>
python LOSA-DNN.py     
</pre>
or you can check the python notebook file inside the notebook folder.

