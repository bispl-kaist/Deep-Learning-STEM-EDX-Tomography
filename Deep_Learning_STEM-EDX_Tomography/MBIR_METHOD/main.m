clear ;

restoredefaultpath();

addpath('./lib_optim');
addpath('./tomo_func');

%% Data directories
input_dir = '../Denoise_CNN/result/denoising/';

output_dir = '../SR_CNN/data/sr/test/';
if ~isdir(output_dir)
    mkdir(output_dir);
end

result_dir = './result/l1/';
if ~isdir(result_dir)
    mkdir(result_dir);
end

input_lst = dir([input_dir 'output_*.mat']);

%% CT system
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

% %% Rotation Setup
nNumView_syn    = 360;                       % [elements]
dStepView_syn   = 2*pi/nNumView_syn;              % [radian]

% %% Make Object (Image, Detector)
param_syn       = MakeParam(pdImgSize, pnImgSize, pdStepDct, pnSizeDct, pdOffsetDct, dStepView_syn, nNumView_syn, DSO, DSD);

% %% Computed Tomography (CT) Operators
A               = @(x) Projection(x, param);
AT              = @(y) BackProjection(y, param)/(pi/(2*nNumView));

% %% Synthetic Computed Tomography (CT) Operators
A_syn           = @(x) Projection(x, param_syn);

%% Conjugate Gradient method
% %% Conjugate Gradient (CG) method's Hyperparameters
lambda1         = 1e2;
mu1             = 1e2;

niter          	= 3e1;
ninner          = 3e1;

wgt             = 400;
scale           = 1e3;

set_atom        = {'se', 's', 'zn'};

% %% Model-based Iterative Reconstruction (MBIR) Operators
A0              = @(x, y)	(AT(A(x) - y) + mu1*(b - d + x));
A1              = @(x)   	(AT(A(ones(size(x), 'single'))) + mu1);

Acg             = @(x)      (AT(A(x)) + mu1*x);

L2              = @(x) 1/2*sqrt(sum(x(:).^2));
L1              = @(x) sum(abs(x(:)));

COST.equation   = '1/2 * || A(X) - Y ||_2^2 + lambda1 * | X |';
COST.function	= @(x, y) 1/2 * L2(A(x) - y) + lambda1 * L1(x);

%% Run Reconstruction
for iobj = 1:nNumView:length(input_lst)
    prj = zeros(param.nDctX, param.nDctY, length(set_atom), param.nNumView, 'single');
    
    % Sinogram Centering & Cleaning
    load(num2str(fix(iobj/nNumView), 'param_obj%d.mat'));
    
    for iview = 1:nNumView
        load([input_dir input_lst((iobj - 1) + iview).name]);
        
        output  = imrotate(output, rot_img_set(iview), 'bilinear','crop');
        output  = circshift(output, circ_x_img_set(iview), 2);
        output  = circshift(output, circ_y_img_set(iview), 1);
        output 	= bsxfun(@times, output, msk(:,:,iview));
        
        prj(:, :, :, iview) = flip(permute(output, [2, 1, 3]), 1);
    end
    
    for iatom = 1:length(set_atom)
        name_atom       = set_atom{iatom};
        y               = squeeze(prj(:,:,iatom,:));
        y               = wgt*y;
        
        y(y < 0)        = 0;
        
        %% Synthetic data
        x              	= zeros(pnImgSize, 'single');
        x_              = [];
        y_              = [];
        
        dk              = zeros(pnImgSize, 'single');
        
        bk              = zeros(pnImgSize, 'single');
        obj             = zeros(niter, 1);
        
        ATy             = AT(y);
        
        %% Run CG method
        for iter = 1:niter
            b0          = ATy + mu1*(dk - bk);
            x           = CG(Acg, b0, x, ninner);
            x(x < 0)    = 0;
            
            bufk        = (x) + bk;
            dk          = shrink(bufk, lambda1/mu1);
            bk          = bufk - dk;
            
            obj(iter)   = COST.function(x, y);
            
            figure(iatom); colormap gray;
            suptitle([num2str(iter) ' / ' num2str(niter) ': ' name_atom]);
            subplot(2,2,1);     imagesc(squeeze(x(:,:,floor(end/2))));      title('Axial-view');
            subplot(2,2,2);     imagesc(squeeze(x(floor(end/2), :, :)));    title('Coronal-view');
            subplot(2,2,[3,4]); plot(obj(1:iter));                      	title(['Cost Func.: ' COST.equation]);
            
            drawnow();
        end
        
        x   = x./wgt;
        
        save([result_dir 'recon_' num2str(fix(iobj/nNumView), '%04d_') name_atom '.mat'], 'x'); 
    end
end

%% Make Systhetic sinogram
for iobj = 1:nNumView:length(input_lst)
    
    y = [];
    
    for iatom = 1:length(set_atom)
        name_atom       = set_atom{iatom};
        
        load([result_dir 'recon_' num2str(fix(iobj/nNumView), '%04d_') name_atom '.mat'], 'x'); 
        
%         y_              = A_syn(flip(x, 1));
        y_              = A_syn(x);
        y_(y_ < 0)      = 0;
        y_              = permute(y_, [2, 1, 4, 3]);
        y               = cat(3, y, y_);
    end
    
    for iview = 1:nNumView_syn
        input = permute(y(:,:,:,iview), [2,1,3]);
        save([output_dir 'input_' num2str(fix(iobj/nNumView)*nNumView_syn + iview - 1, '%04d') '.mat'], 'input'); 
    end
end