clear ;

reset(gpuDevice(1));

restoredefaultpath();
addpath('./lib_util');
addpath('./lib_optim');
addpath('./tomo_func');

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

nNumView        = 360;                       % [elements]
dStepView       = 2*pi/nNumView;              % [radian]

% %% Make Object (Image, Detector)
param           = MakeParam(pdImgSize, pnImgSize, pdStepDct, pnSizeDct, pdOffsetDct, dStepView, nNumView, DSO, DSD);

%% Definition of Operators
A               = @(x) Projection(x, param);
AT              = @(y) BackProjection(y, param)/(pi/(2*nNumView));
ATA             = @(x) AT(A(x));

%%
dataType    = 'test';
smp         = 30;

%%
navg        = 5;
npatch      = 256;
nch         = 3;

kerType     = 'cnn';
nid_set     = 1:12;
dataName    = 'sait_tomo';
outputName	= 'sait_recon_full';

wgt         = 400;

for nepoch      = 30

nsmp        = 30;
nettype_set     = {'cgan'};

for inettype = 1:length(nettype_set)
    
    nettype	= nettype_set{inettype};

for nid         = nid_set
    
    dir_output = ['./' outputName '/' nettype '/avg' num2str(navg) '/epoch' num2str(nepoch) '/' num2str(nid) '/test'];
    
    [st_, m_, m__] = mkdir(dir_output);
    dir_result	= ['./results/sait/' nettype '/epoch' num2str(nepoch) '/avg' num2str(navg) '/' kerType '/' num2str(nid) '/'];
    
    %% Load data
    imdb_se = load([dir_result 'se_l1_cg.mat']);
    imdb_s  = load([dir_result 's_l1_cg.mat']);
    imdb_zn = load([dir_result 'zn_l1_cg.mat']);
    
    y_se    = A(imdb_se.x);
    y_s     = A(imdb_s.x);
    y_zn    = A(imdb_zn.x);
    
    labels  = repmat(cat(3, permute(y_se, [1,2,4,3]), permute(y_s, [1,2,4,3]), permute(y_zn, [1,2,4,3])), [1,1,1,1,length(smp)]);
    data    = labels;
    
    labels  = reshape(labels, 256, 256, 3, []);
    data    = reshape(data, 256, 256, 3, []);
        
    itest           = -1;

    for iview = 1:nNumView
        label   = labels(:,:,:,iview);
        input   = data(:,:,:,iview);
        
        img     = cat(2, label, input);
        
        figure(1); imagesc(img);
        title(max(img(:))); drawnow();
        
        itest = itest + 1;
        idx = itest;
        
        save([dir_output '/input_' num2str(idx, '%04d') '.mat'], 'input', '-v6');
        save([dir_output '/label_' num2str(idx, '%04d') '.mat'], 'label', '-v6');
    end
end
end
end

return ;
