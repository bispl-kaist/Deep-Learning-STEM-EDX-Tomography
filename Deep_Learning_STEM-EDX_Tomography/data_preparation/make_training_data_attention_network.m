clear; clc;
addpath('lib');
sz_ker      = 3;
ker         = fspecial('average', sz_ker);
no_random   = 30; % the number of bootstrap subsampling
imgSize     = 256;

se_reg      = zeros(imgSize, imgSize, no_random);
s_reg       = zeros(imgSize, imgSize, no_random);
zn_reg      = zeros(imgSize, imgSize, no_random);
%% Se 
dataDir_se          = 'regression_data\1st_atom\Result\SAIT'; % the directory of result of regression network for se
saveDir_se          = 'attention_data\1st_atom\Input\SAIT'; % the directory of input of attention network for se
th_se               = 0.02;

[se_input, se_label]= mask_data(dataDir_se, saveDir_se, 'sei', no_random, th_se, se_reg);

%% S
dataDir_s           = 'regression_data\2nd_atom\Result\SAIT'; % the directory of result of regression network for s
saveDir_s           = 'attention_data\2nd_atom\Input\SAIT'; % the directory of input of attention network for s
th_s                = 0.02;

[s_input, s_label]  = mask_data(dataDir_s, saveDir_s, 'si', no_random, th_s, se_reg);

%% Zn
dataDir_zn          = 'regression_data\3rd_atom\Result\SAIT'; % the directory of result of regression network for zn
saveDir_zn          = 'attention_data\3rd_atom\Input\SAIT'; % the directory of input of attention network for zn
th_zn               = 0.02;

[zn_input, zn_label]= mask_data(dataDir_zn, saveDir_zn, 'zni', no_random, th_zn, se_reg);
