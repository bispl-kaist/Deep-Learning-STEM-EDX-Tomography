clear; clc;

addpath('./utils');

%% System parameters
% %% Parameter Setting
DSO             = 400;                      % [mm]
DSD             = 1000;                     % [mm]

% %% Make Object
pdImgSize       = [256, 256, 256];          % [mm x mm]
pnImgSize       = [256, 256, 256];

% %% Make Detector
pdStepDct     	= 1;                        % [mm]
pnSizeDct       = [pnImgSize(3), 256];      % [elements]

pdOffsetDct     = 0;                        % [elements]

% %% Rotation Setup
nNumView        = 3;                       % [elements]
dStepView       = 2*pi/360*40;              % [radian]

% %% Make Object (Image, Detector)
param           = MakeParam(pdImgSize, pnImgSize, pdStepDct, pnSizeDct, pdOffsetDct, dStepView, nNumView, DSO, DSD);


%% Definition of Operators
A               = @(x) Projection(x, param);
AT              = @(y) BackProjection(y, param)/(pi/(2*nNumView));
AINV            = @(y) BackProjection(Filtering(y, param), param);
ATA             = @(x) AT(A(x));

%% Filtered BackProjection
wgt = 400;
specimen = 'sQD2/';
root = ['../../EDX_measurements/' specimen];
save_dir = '../datasets/test/';
mkdir(save_dir);
atoms = {'se', 's', 'zn'};

load([root 'EDX_recon_w_CNN']);

y_se = flip(permute(single(imdb.se_outputs_cnn), [2, 1, 3]), 1);
y_s  = flip(permute(single(imdb.s_outputs_cnn), [2, 1, 3]), 1);
y_zn = flip(permute(single(imdb.zn_outputs_cnn), [2, 1, 3]), 1);

th          = 0;
y_se        = wgt*y_se;
y_s         = wgt*y_s;
y_zn        = wgt*y_zn;
y_se(y_se < th)   = 0;
y_s(y_s < th)     = 0;
y_zn(y_zn < th)   = 0;

x_se = AINV(y_se);
x_s = AINV(y_s);
x_zn = AINV(y_zn);

x_se = x_se./wgt;
x_s = x_s./wgt;
x_zn = x_zn./wgt;

x_se(x_se < 0) = 0;
x_s(x_s < 0) = 0;
x_zn(x_zn < 0) = 0;

x_3chan = cat(4, x_se, x_s, x_zn);

save([save_dir 'FBP_tomo' num2str(datai)], 'x_3chan');