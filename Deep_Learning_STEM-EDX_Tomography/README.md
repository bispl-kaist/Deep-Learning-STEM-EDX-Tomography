# Deep learning STEM-EDX tomography of nanocrystals

We provide tensorflow(python) and matlab implementations of **Deep learning STEM-EDX tomography
of nanocrystals**. This code was written by **Eunju Cha** and **Yoseob Han**.


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

---

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

### Getting started (Test)

**1) Download the pretrained networks**
```
./install.m
```
Run the matlab script for downloading pretrained networks such as **Denoising network** and **Projection Enhancement network**. 


**2) Denoising network**
```
./data_preparation/make_testing_data_denoising.m
```
Run the matlab script is to make the testing data.

```
/Denoise_CNN/main.py
```
The above python script is the main file to be executed for both training and testing of
Denoising network. Training and testing of the model can be done with
```
# Train Denoise CNN
./DenoiseCNN_train.sh

# Test Denoise CNN
./DenoiseCNN_test.sh
```

Specific options including load/save directions can be modified inside bash scripts.

**Inference time:** About 0.35 sec / slice

**3) Model-Based Iterative Reconstruction (MBIR)**
```
./MBIR_METHOD/main.m
```
Run the matlab script for executing model-based iterative reconstruction with conjugate gradient (CG)
method. Upon running the code you will make both 3D-reconstructed data and its projection data.

**Inference time:** About 300 sec / object

**4) Projection Enhancement network**
```
./SR_CNN/main.py
```
Similar to denoising network, this is the main file to be executed for both training and testing of
Projection Enhancement network. Training and testing of the model can be done with
```
# Train SR CNN
./SRNN_train.sh

# Test SR CNN
./SRCNN_test.sh
```
Options including load/save directions can be modified inside bash scripts.

**Inference time:** About 0.35 sec / slice

**5) Analytic Reconstruction using Filtered Back-Projection (FBP)**
```
./main.m
```
Run the matlab script for executing FBP to reconstruct final result. 

**Inference time:** About 3.55 sec / object

**6) Reproduce Figure. 4(a) and 4(b)**
```
./run_fig4a.m
./run_fig4b.m
```
Run the above matlab scripts for reproducing the **figure. 4(a)** and **4(b)** in the paper.

---

# Analysis part
Last part of our work for 3D tomography. We provide easy analysis method for 3D reconstruction results. Each reconstruction data of nano-particles (quantum dot) has information on 3 elements (S, Se, Zn), and all of them are mapped in 2D maps using Cartesian to Spherical transformation, and then we compensate distortion along longitude direction with Matlab. It is similar with Mollweide projection of pseudocylindrical map projection.


### Prerequisites
- Matlab >= R2019b

### Environments
The analysis software is tested on Windows operating systems. The developmental version of the package has been tested on the following systems and drivers

- Windows 10 64bit

### Getting started

**1) 2D Projection Data**

```
#Generated 2-D maps for SQD1-1

8_spherical_projection_particle_1_s_se_zn_zn(se)_zn(s)_inter_linear_step_1_thr_0.003.mat

#Generated 2-D maps for SQD1-2

8_spherical_projection_particle_2_s_se_zn_zn(se)_zn(s)_inter_linear_step_1_thr_0.003.mat

#Generated 2-D maps for SQD2-1

12_spherical_projection_particle_1_s_se_zn_zn(se)_zn(s)_inter_linear_step_1_thr_0.003.mat

#Generated 2-D maps for SQD2-2

12_spherical_projection_particle_2_rot_70_s_se_zn_zn(se)_zn(s)_inter_linear_step_1_thr_0.003.mat
```
 
The variables of thickness_xx_mod are finally compensated 2D maps for each elements.


**2) Display code**

```
#Display thickness maps and histograms for SQD1-1

./Analysis/display code for SQD1_1.m

#Display thickness maps and histograms for SQD1-2

./Analysis/display code for SQD1_2.m

#Display thickness maps and histograms for SQD2-1

./Analysis/display code for SQD2_1.m

#Display thickness maps and histograms for SQD2-2

./Analysis/display code for SQD2_2.m
```

Run the above matlab scripts for reproducing the Supplementary Figures S17-20 and Supplementary Table S1 in the paper. 

---

# 3D Reconstruction Data

We provide reconstruction results of all cQD, sQD1, sQD2 shown in the manuscript in ```.raw``` file format so that it is possible for the interested readers to inspect the data on their own. All the data are contained in the folder ```./Recon_data```, with size ```140 x 140 x 256``` with ```uint16``` precision. 
