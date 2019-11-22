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


for nepoch      = 30

nettype_set     = {'cgan'};

for inettype = 1:length(nettype_set)
    
    nettype	= nettype_set{inettype};
    
for navg        = 5
npatch      = 256;
nch         = 3;

kerType_set = {'cnn', 'std'};

for ikerType     = 1:length(kerType_set)        % 'cnn', 'std'
    
kerType     = kerType_set{ikerType};
nid_set = 1:12;
dataName    = 'sait_tomo';

% CHANGE THIS PART SO THAT DATA IS READABLE
% CHANGE THIS PART SO THAT DATA IS READABLE
% CHANGE THIS PART SO THAT DATA IS READABLE
dir_data	= ['./data/sait/' nettype '/avg' num2str(navg) '/epoch' num2str(nepoch) '/'];

atom_set    = {'se', 's', 'zn'};
    
wgt         = 400;
scale       = 1e3;

for nid         = nid_set    
    % CHANGE THIS PART SO THAT DATA IS SAVABLE
    % CHANGE THIS PART SO THAT DATA IS SAVABLE
    % CHANGE THIS PART SO THAT DATA IS SAVABLE
    dir_result	= ['./results/sait/' nettype '/epoch' num2str(nepoch) '/avg' num2str(navg) '/' kerType '/' num2str(nid) '/'];
    mkdir(dir_result);
    
    for iatom = 1:length(atom_set)
        atomName    = atom_set{iatom};
        % CHANGE THIS PART SO THAT DATA IS READABLE
        % CHANGE THIS PART SO THAT DATA IS READABLE
        % CHANGE THIS PART SO THAT DATA IS READABLE
        imdb        = load([dir_data 'imdb_' dataName num2str(nid) '.mat']);
        
        if strcmp(kerType, 'cnn')
            switch atomName
                case 'se'
                    y = flip(permute(single(imdb.se_outputs_cnn), [2, 1, 3]), 1);
                case 's'
                    y = flip(permute(single(imdb.s_outputs_cnn), [2, 1, 3]), 1);
                case 'zn'
                    y = flip(permute(single(imdb.zn_outputs_cnn), [2, 1, 3]), 1);
            end
        else
            switch atomName
                case 'se'
                    y = flip(permute(single(imdb.se_outputs_std), [2, 1, 3]), 1);
                case 's'
                    y = flip(permute(single(imdb.s_outputs_std), [2, 1, 3]), 1);
                case 'zn'
                    y = flip(permute(single(imdb.zn_outputs_std), [2, 1, 3]), 1);
            end
        end
        
        y           = wgt*y;
        
        th          = 0;
        y(y < th)   = 0;
        
        %% Hyper parameters
        lambda1         = 1e2;
        mu1             = 1e2;
        
        niter          	= 3e1;
        ninner          = 3e1;
        
        x_iter          = [1, 3, 5];
        
        %% Definition of Operators
        A               = @(x) Projection(x, param);
        AT              = @(y) BackProjection(y, param)/(pi/(2*nNumView));
        AINV            = @(y) BackProjection(Filtering(y, param), param);
        ATA             = @(x) AT(A(x));
        
        %%
        A0              = @(x, y)	(AT(A(x) - y) + mu1*(b - d + x));
        A1              = @(x)   	(AT(A(ones(size(x), 'single'))) + mu1);
        
        Acg             = @(x)      (AT(A(x)) + mu1*x);
        
        L2              = @(x) 1/2*sqrt(sum(x(:).^2));
        L1              = @(x) sum(abs(x(:)));
        
        COST.equation   = '1/2 * || A(X) - Y ||_2^2 + lambda1 * | X |';
        COST.function	= @(x) 1/2 * L2(A(x) - y) + lambda1 * L1(x);
        
        %% Synthetic data
        x0              = AINV(y);
        x              	= zeros(pnImgSize, 'single');
        x_              = [];
        y_              = [];
        
        dk               = zeros(pnImgSize, 'single');
        
        bk              = zeros(pnImgSize, 'single');
        obj             = zeros(niter, 1);
        
        ATy     = AT(y);
        %% Compressed Sensing rootine
        for iter = 1:niter
            
            b0              = ATy + mu1*(dk - bk);
            x               = CG(Acg, b0, x, ninner);
            x(x < 0)        = 0;
            
            x__             = x;
            x__(x__ < 0)    = 0;
            y__             = A(x);
            y__(y__ < 0)   = 0;
            
            if sum(find(iter == x_iter)) ~= 0
                x_              = cat(4, x_, x__);
            end
            
            y_              = cat(4, y_, y__);
            
            bufk        = (x) + bk;
            dk          = shrink(bufk, lambda1/mu1);
            bk          = bufk - dk;
            
            obj(iter)  = COST.function(x);
            
            figure(nid*10 + iatom); colormap gray;
            
            subplot(3,4,1);   imagesc(squeeze(y(:,:,1)));
            subplot(3,4,2);   imagesc(squeeze(y(:,:,4)));
            subplot(3,4,3);   imagesc(squeeze(y(:,:,7)));
            subplot(3,4,4);   imagesc(squeeze(y(:,:,10)));
            
            subplot(3,4,5);   imagesc(squeeze(y__(:,:,1)));
            subplot(3,4,6);   imagesc(squeeze(y__(:,:,4)));
            subplot(3,4,7);   imagesc(squeeze(y__(:,:,7)));
            subplot(3,4,8);   imagesc(squeeze(y__(:,:,10)));
            
            subplot(3,4,9);     imagesc(squeeze(x(:,:,floor(end/2))));      title([num2str(iter) ' / ' num2str(niter)]);
            subplot(3,4,10);    imagesc(squeeze(x(floor(end/2), :, :)));
            subplot(3,4,11);    plot(obj(1:iter));                            title(COST.equation);
            
            drawnow();
        end
        
        x   = x./wgt;
        y   = y./wgt;
        y_  = y_./wgt;
        
        vtkwrite([dir_result atomName '_l1_cg.vtk'], 'structured_points', atomName, scale*x);
        save([dir_result atomName '_l1_cg.mat'], 'x', 'x_', 'x_iter');
        save([dir_result atomName '_proj.mat'], 'y', 'y_');
        
    end
end
end
end
end
end
return ;