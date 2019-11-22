%% Âü°í¹®Çå
% https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique

%% ART Equation
% x^(k+1) = x^k + lambda * AT(b - A(x))/ATA 

%%
clear ;

isgpu   = true;

%%  SYSTEM SETTING
N       = 512;
VIEW    = 30;
THETA   = linspace(0, 180, VIEW + 1);   THETA(end) = [];

%% DATA GENERATION
load('XCAT512.mat');
x       = imresize(double(XCAT512), [N, N]);
p       = radon(x, THETA);
x_full  = iradon(p, THETA, N);

%% LOW-DOSE SINOGRAM GENERATION
i0     	= 1e4;
pn     	= exp(-p);
pn     	= i0.*pn;
pn     	= poissrnd(pn);
pn      = max(-log(max(pn,1)./i0),0);

%% Algebraic Reconstruction Technique (ART) INITIALIZATION
x_low   = iradon(pn, THETA, N);
x0      = zeros(size(x));
lambda  = 1e0;
niter   = 2e2;

A       = @(x) radon(x, THETA);
AT      = @(p) iradon(p, THETA, 'none', N);

%% RUN Algebraic Reconstruction Technique (ART)
if isgpu
    pn = gpuArray(pn);
    x0 = gpuArray(x0);
end

x_art   = ART(A, AT, pn, x0, lambda, niter);

%% CALCUATE QUANTIFICATION FACTOR 
x_low       = max(x_low, 0);
x_art       = max(x_art, 0);
nor         = max(x(:));

mse_x_low   = immse(x_low./nor, x./nor);
mse_x_art   = immse(x_art./nor, x./nor);

psnr_x_low 	= psnr(x_low./nor, x./nor);
psnr_x_art 	= psnr(x_art./nor, x./nor);

ssim_x_low  = ssim(x_low./nor, x./nor);
ssim_x_art  = ssim(x_art./nor, x./nor);


%% DISPLAY
wndImg  = [0, 0.03];

figure(1); 
colormap(gray(256));

suptitle('Algebraic Reconstruction Technique');
subplot(221);   imagesc(x,     	wndImg); 	axis image off;     title('ground truth');
subplot(222);   imagesc(x_full, wndImg);   	axis image off;     title(['full-dose_{view : ', num2str(VIEW) '}']);
subplot(223);   imagesc(x_low,  wndImg);   	axis image off;     title({['low-dose_{view : ', num2str(VIEW) '}'], ['MSE : ' num2str(mse_x_low, '%.4e')], ['PSNR : ' num2str(psnr_x_low, '%.4f')], ['SSIM : ' num2str(ssim_x_low, '%.4f')]});
subplot(224);   imagesc(x_art,  wndImg);  	axis image off;     title({['recon_{art}'], ['MSE : ' num2str(mse_x_art, '%.4e')], ['PSNR : ' num2str(psnr_x_art, '%.4f')], ['SSIM : ' num2str(ssim_x_art, '%.4f')]});

