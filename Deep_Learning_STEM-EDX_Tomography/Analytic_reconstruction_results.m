clear ;

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

% %% Rotation Setup
nNumView        = 360;                       % [elements]
dStepView       = 2*pi/nNumView;             % [radian]

% %% Make Object (Image, Detector)
param           = MakeParam(pdImgSize, pnImgSize, pdStepDct, pnSizeDct, pdOffsetDct, dStepView, nNumView, DSO, DSD);

filter          = 'hann';

%% Definition of Operators
A               = @(x) Projection(x, param);
AT              = @(y) BackProjection(y, param)/(pi/(2*nNumView));
ATA             = @(x) AT(A(x));
AINV            = @(y) BackProjection(filterProjections(y, filter), param);

%%
dataType    = 'test';

%%
for navg        = 5
npatch      = 256;
nch         = 3;

kerType     = 'cnn';
nid_set     = 1:12;
dataName    = 'sait_tomo';
outputName	= 'sait_recon_full';
net_type_set = {'cgan'};

wgt         = 400;
nrecon      = 256;

for nepoch      = 30

for inet    = 1:length(net_type_set)

net_type    = net_type_set{inet};

th          = 0.0;

for nid         = nid_set    
    disp(nid);
    dir_result	= ['../results/' outputName '/' net_type '/avg' num2str(navg) '/epoch' num2str(nepoch) '/' num2str(nid) '/'];
    mkdir(dir_result);
    
    dir_input 	= [outputName '/' net_type '/avg' num2str(navg) '/epoch' num2str(nepoch) '/' num2str(nid) '/test/'];
    
    P           = zeros([pnSizeDct, 3, nNumView], 'single');
    
    for iview = 1:nNumView
        load(['./' dir_input 'output_' num2str(iview - 1, '%04d') '.mat'], 'output');
        P_  = output;
        
        P_(P_ < th) = 0;
        P(:,:,:,iview) = wgt.*P_;
        
        P_se    = P_;  P_se(:, :, [2, 3])  = 0;
        P_s     = P_;  P_s(:, :, [1, 3])   = 0;
        P_zn    = P_;  P_zn(:, :, [1, 2])  = 0;
        
        figure(1);
        subplot(141);   imagesc(P_); title(iview);
        subplot(142);   imagesc(P_se(:,:,1)); title(iview);
        subplot(143);   imagesc(P_s(:,:,2)); title(iview);
        subplot(144);   imagesc(P_zn(:,:,3)); title(iview);
        
        drawnow();
        
    end
    
    x_se	= AINV(squeeze(P(:,:,1,:)))./wgt;    x_se(x_se < 0)  = 0;
    x_s     = AINV(squeeze(P(:,:,2,:)))./wgt;    x_s(x_s < 0)    = 0;
    x_zn    = AINV(squeeze(P(:,:,3,:)))./wgt;    x_zn(x_zn < 0)  = 0;
    
    vtkwrite([dir_result 'se_proposed_rec' num2str(nrecon) '.vtk'], 'structured_points', 'se', 1e3*x_se);
    vtkwrite([dir_result 's_proposed_rec' num2str(nrecon) '.vtk'], 'structured_points', 's', 1e3*x_s);
    vtkwrite([dir_result 'zn_proposed_rec' num2str(nrecon) '.vtk'], 'structured_points', 'zn', 1e3*x_zn);
    
    x = x_se;   save([dir_result 'se_proposed_rec' num2str(nrecon) '.mat'], 'x');
    x = x_s;    save([dir_result 's_proposed_rec' num2str(nrecon) '.mat'], 'x');
    x = x_zn;	save([dir_result 'zn_proposed_rec' num2str(nrecon) '.mat'], 'x');
end
end
end
end

return ;