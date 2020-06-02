clear ;

restoredefaultpath();

addpath('./MBIR_METHOD/lib_optim');
addpath('./MBIR_METHOD/tomo_func');

%% Step. 1 Download the pretrained networks
% Run the matlab script is to download the pretrained networks such as
% Denoising CNN and Super-resolution CNN
%
% ./install.m

%% Step. 2 Test dataset Generation
% Run the matlab script is to make the testing data for Denoising CNN
%
% './data_preparation/make_testing_data_denoising_cnn.m'

%% Step. 3 Inference Denosing CNN
% bash ./Denoise_CNN/DenoiseCNN_test.sh

%% Step. 4 Run Model-Based Iterative Reconstruction (MBIR) method
% ./MBIR_METHOD/main.m

%% Step. 5 Inference Super Resolution CNN
% bash ./SR_CNN/SRCNN_test.sh

%% Step. 6 Analytic Reconstruction using Filtered Back-Projection (FBP)
% %% Data directories
input_dir = './SR_CNN/result/sr/';

result_dir = './result/proposed/';
if ~isdir(result_dir)
    mkdir(result_dir);
end

input_lst = dir([input_dir '*.mat']);

% %% CT system
% Parameter Setting
DSO             = 400;                      % [mm]
DSD             = 1000;                     % [mm]

% Make Object
pdImgSize       = [256, 256, 256];          % [mm x mm]
pnImgSize       = [256, 256, 256];

% Make Detector
pdStepDct     	= 1;                        % [mm]
pnSizeDct       = [pnImgSize(3), 256];      % [elements]

pdOffsetDct     = 0;                        % [elements]

% Rotation Setup
nNumView        = 360;                       % [elements]
dStepView       = 2*pi/nNumView;              % [radian]

% Make Object (Image, Detector)
param           = MakeParam(pdImgSize, pnImgSize, pdStepDct, pnSizeDct, pdOffsetDct, dStepView, nNumView, DSO, DSD);

% Computed Tomography (CT) Operators
AINV            = @(y) BackProjection(filterProjections(y, 'hann'), param);

% %% FBP Reconstruction
set_atom        = {'se', 's', 'zn'};
tic;
for iobj = 1:nNumView:length(input_lst)
    prj = zeros(param.nDctX, param.nDctY, length(set_atom), param.nNumView, 'single');
    
    for iview = 1:nNumView
        load([input_dir input_lst((iobj - 1) + iview).name]);
        prj(:, :, :, iview) = output;
    end
    
    prj   = circshift(prj, 210, 4);
    
    for iatom = 1:length(set_atom)
        name_atom       = set_atom{iatom};
        y               = squeeze(prj(:,:,iatom,:));
        y(y < 0)        = 0;
        
        x               = AINV(y);
        x(x < 0)        = 0;
        
        save([result_dir 'recon_' num2str(fix(iobj/nNumView), '%04d_') name_atom '.mat'], 'x'); 
    end
end
toc;
