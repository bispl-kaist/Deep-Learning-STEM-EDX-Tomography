# Deep learning STEM-EDX tomography of nanocrystals

We provide tensorflow(python) and matlab implementations of **Deep learning STEM-EDX tomography
of nanocrystals. This code was written by **Eunju Cha** and **Yoseob Han**.


### Datasets
```
# Test dataset for CNN-based kernel
./data_preparation/data/CNN-based kernel

# Test dataset for 3D tomography
./data_preparation/data/3D tomography
```
Dataset for inference is contained in the above folders.


First part of our work for **CNN-based kernel** consists mainly of two parts - **Regression network** and **Attention module**.
We provide both training and inference codes for these deep neural networks, respectively.

### Prerequisites
- Tensorflow >= 1.12.0
- numpy == 1.17.0
- pillow == 6.1.0
- _pickle

### Environments
The package development version is tested on Linux operating systems. The developmental version of
the package has been tested on the following systems and drivers

- Linux 16.04
- CUDA 10.1

### Getting started

**1) Regression network**

```
./data_preparation/make_training_data_regression_network.m

```
Run the matlab script is to make the training data for regression network by exploiting bootstrap subsampling.

```
./Regression_network/main.py
```
The above python script is the main file to be executed for both training and testing of
regression network. Training and testing of the model can be done with

```
# Train regression newtork
./Regression_network/scripts/regression_train.sh
./Regression_network/scripts/regression_train_concat.sh

# Test regression network
./Regression_network/scripts/regression_test.sh
./Regression_network/scripts/regression_test_concat.sh
```
Options including load/save directions can be modified inside bash scripts.

**2) Attention module**
```
./data_preparation/make_training_data_regression_network.m
```
Run the matlab script is to make the training data for attention module which is used to effeciently aggregated the entire output of the regression network.

```
./Attention_network/main.py
```
This is the main file to be executed for both training and testing of attention module. Training and testing of the model can be done with
```
# Train attention module
./Attention_network/scripts/attention_train.sh

# Test attention module
./Regression_network/scripts/attention_test.sh
```
Specific options including load/save directions can be modified inside bash scripts.


Second part of our work for **3D tomography** consists mainly of two parts - **Denoising CNN** and **Super-resolution CNN**.
We provide both training and inference codes for these deep neural networks, respectively.
Furthermore, model-based iterative reconstruction using conjugate gradient method, along with
 analytical reconstruction of the tomography was implemented with matlab.

### Prerequisites
- Tensorflow >= 1.12.0
- tqdm
- numpy == 1.17.0
- pillow == 6.1.0

### Environments
The package development version is tested on Linux operating systems. The developmental version of
the package has been tested on the following systems and drivers

- Linux 18.04
- CUDA 10.1

### Getting started

**1) Denoising CNN**

```
/Denoise_CNN/main_sait.py
```
The above python script is the main file to be executed for both training and testing of
Denoising CNN. Training and testing of the model can be done with
```
# Train Denoise CNN
./DenoiseCNN_train.sh

# Test Denoise CNN
./DenoiseCNN_test.sh
```

Specific options including load/save directions can be modified inside bash scripts.

**2) Model-based iterative reconstruction**
```
MIBR.m
```
Run the matlab script for executing model-based iterative reconstruction with conjugate gradient
method. Upon running the code you will make both 3D-reconstructed data and its projection data.

**3) Projection**
```
./data_preparation/Make_projection_db_seperate.m
./data_preparation/Make_projection_db_structured.m
```
Use either one of the scripts to create data for training/testing Super-resolution.
The former will save each files seperately, whereas the latter will save all the data into one
structured one.

**4) Super-resolution CNN**
```
./SR_CNN/main_sait.py
```
Similar to denoising CNN, this is the main file to be executed for both training and testing of
Denoising CNN. Training and testing of the model can be done with
```
# Train Denoise CNN
./DenoiseCNN_train.sh

# Test Denoise CNN
./DenoiseCNN_test.sh
```
Options including load/save directions can be modified inside bash scripts.



