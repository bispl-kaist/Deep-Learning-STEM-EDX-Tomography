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
nNumView        = 13;                       % [elements]
dStepView       = 2*pi/360*10;              % [radian]

% %% Make Object (Image, Detector)
param           = MakeParam(pdImgSize, pnImgSize, pdStepDct, pnSizeDct, pdOffsetDct, dStepView, nNumView, DSO, DSD);

        
%% Definition of Operators
A               = @(x) Projection(x, param);
AT              = @(y) BackProjection(y, param)/(pi/(2*nNumView));
AINV            = @(y) BackProjection(Filtering(y, param), param);
ATA             = @(x) AT(A(x));


for navg        = 5
npatch      = 256;
nch         = 3;
nepoch      = 30;

nid_set     = 1:12;
method = 'cgan';
dataName    = 'sait';
outputName  = 'sait_recon';
dir_data	= ['./data/sait/' method num2str(npatch) '/ch' num2str(nch) '/avg' num2str(navg)  '/'];

smp         = 30;

%%
labels      = zeros(pnImgSize(1), pnImgSize(2), 3, length(smp)*nNumView*length(nid_set), 'single');
data        = zeros(pnImgSize(1), pnImgSize(2), 3, length(smp)*nNumView*length(nid_set), 'single');
set_id      = zeros(1, length(smp)*nNumView*length(nid_set), 'single');
set_avg     = zeros(1, length(smp)*nNumView*length(nid_set), 'single');

cnt         = 0;


%%
trnType     = 'all';        % 'all', 'each'
kerType     = 'cnn';        % 'cnn', 'std'

figure(1);
set(gca,'position',[0 0 1 1]);
set(gcf,'PaperPositionMode','auto');

figure(2);
set(gca,'position',[0 0 1 1]);
set(gcf,'PaperPositionMode','auto');

for nid         = nid_set
    
    dir_result	= ['./results/' dataName '/cgan' '/epoch' num2str(nepoch) '/avg' num2str(navg) '/' kerType '/' num2str(nid) '/'];
    
    %% Load data
    proj_se = load([dir_result 'se_proj.mat']);
    proj_s  = load([dir_result 's_proj.mat']);
    proj_zn = load([dir_result 'zn_proj.mat']);
    
    rec_se  = load([dir_result 'se_l1_cg.mat'], 'x_');
    rec_s   = load([dir_result 's_l1_cg.mat'], 'x_');
    rec_zn  = load([dir_result 'zn_l1_cg.mat'], 'x_');
    
    y_se_   = [];
    y_s_    = [];
    y_zn_   = [];
    
    
    for i = 1:length(smp)-1
        x_se    = rec_se.x_(:,:,:,i);
        x_s     = rec_s.x_(:,:,:,i);
        x_zn    = rec_zn.x_(:,:,:,i);
        
        x_se(x_se < 1)      = 0;
        x_s(x_s < 0.7)      = 0;
        x_zn(x_zn < 0.8)      = 0;
        
        y_se__  = A(x_se)./400;
        y_s__ 	= A(x_s)./400;
        y_zn__  = A(x_zn)./400;
        
        figure(1);  colormap gray;
        subplot(341);   imagesc(proj_se.y(:,:,1));
        subplot(342);   imagesc(proj_se.y(:,:,4));
        subplot(343);   imagesc(proj_se.y(:,:,9));
        subplot(344);   imagesc(proj_se.y(:,:,13));
        
        subplot(345);   imagesc(proj_s.y(:,:,1));
        subplot(346);   imagesc(proj_s.y(:,:,4));
        subplot(347);   imagesc(proj_s.y(:,:,9));
        subplot(348);   imagesc(proj_s.y(:,:,13));
        
        subplot(349);       imagesc(proj_zn.y(:,:,1));
        subplot(3,4,10);	imagesc(proj_zn.y(:,:,4));
        subplot(3,4,11);	imagesc(proj_zn.y(:,:,9));
        subplot(3,4,12);	imagesc(proj_zn.y(:,:,13));
        
        
        figure(10);  colormap gray;
        subplot(341);   imagesc(y_se__(:,:,1));
        subplot(342);   imagesc(y_se__(:,:,4));
        subplot(343);   imagesc(y_se__(:,:,9));
        subplot(344);   imagesc(y_se__(:,:,13));
        
        subplot(345);   imagesc(y_s__(:,:,1));
        subplot(346);   imagesc(y_s__(:,:,4));
        subplot(347);   imagesc(y_s__(:,:,9));
        subplot(348);   imagesc(y_s__(:,:,13));
        
        subplot(349);       imagesc(y_zn__(:,:,1));
        subplot(3,4,10);	imagesc(y_zn__(:,:,4));
        subplot(3,4,11);	imagesc(y_zn__(:,:,9));
        subplot(3,4,12);	imagesc(y_zn__(:,:,13));
        
        drawnow();
        
        y_se_   = cat(4, y_se_, y_se__);
        y_s_    = cat(4, y_s_, y_s__);
        y_zn_   = cat(4, y_zn_, y_zn__);
    end
    
    y_se_   = cat(4, y_se_, proj_se.y_(:,:,:,end));
    y_s_    = cat(4, y_s_, proj_s.y_(:,:,:,end));
    y_zn_   = cat(4, y_zn_, proj_zn.y_(:,:,:,end));
    
    label_  = repmat(cat(3, permute(proj_se.y, [1,2,4,3]), permute(proj_s.y, [1,2,4,3]), permute(proj_zn.y, [1,2,4,3])), [1,1,1,1,length(smp)]);
    data_   = cat(3, permute(y_se_, [1,2,5,3,4]), permute(y_s_, [1,2,5,3,4]), permute(y_zn_, [1,2,5,3,4]));
    
    label_  = permute(label_, [2, 1, 3, 4, 5]);
    data_   = permute(data_, [2, 1, 3, 4, 5]);
    
    label_  = reshape(label_, 256, 256, 3, []);
    data_   = reshape(data_, 256, 256, 3, []);
    
    for iview = 1:nNumView
        for ismp = 1:length(smp)
            label__     = label_(:,:,:,iview + nNumView*(ismp - 1));
            data__      = data_(:,:,:,iview + nNumView*(ismp - 1));
            
            
            img     = cat(2, label__, data__);
            figure(1); imagesc(label__);
            figure(2); imagesc(data__);
            title([num2str(max(label__(:))) ' / ' num2str(max(data__(:)))]);
        end
        drawnow();
    end
    
    labels(:,:,:,cnt + (1:nNumView*length(smp))) 	= label_;
    data(:,:,:,cnt + (1:nNumView*length(smp)))  	= data_;
    set_id(cnt + (1:nNumView*length(smp)))      	= nid;
    
    cnt                                             = cnt + nNumView*length(smp);
    
end

images.labels   = labels;
images.data     = data;
images.set_id   = set_id;

save([dir_data 'imdb_' outputName '_smp30.mat'], 'images');
end
return ;