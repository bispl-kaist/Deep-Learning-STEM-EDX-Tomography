%% 
% Âü°í¹®Çå
% https://en.wikipedia.org/wiki/Newton%27s_method

%% 
% COST FUNCTION
% x^* = argmin_x { 1/2*|| A(X) - Y ||_2^2 + lambda * (|| Dx(X) ||_2^2 + || Dy(X) ||_2^2 }
% 
% Newton Method
% x^(k+1) = x^k - f(x^k) / f'(x^k)
%
% s.t.  f(x) = 0;
%       f'(x) = a( f(x) )/ax

%%
clear ;

isgpu   = true;

%%  SYSTEM SETTING
N       = 512;
VIEW    = 360;
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

%% NEWTON METHOD INITIALIZATION
LAMBDA  = 1e1;
A0      = @(x, y)	(iradon(radon(x, THETA) - y, THETA, 'none', N)  + LAMBDA*Dxt(Dx(x)) + LAMBDA*Dyt(Dy(x)));
A1      = @(x)      (iradon(radon(ones(size(x)), THETA), THETA, 'none', N));

x_low   = iradon(pn, THETA, N);
b       = pn;
x0      = zeros(size(x));
% x0      = x_low;
niter   = 2e2;

%% RUN NEWTON METHOD
if isgpu
    b  = gpuArray(b);
    x0 = gpuArray(x0);
end

x_newton        = Newton(A0, A1, b, x0, niter);

%% CALCUATE QUANTIFICATION FACTOR 
x_low           = max(x_low, 0);
x_newton        = max(x_newton, 0);
nor             = max(x(:));

mse_x_low       = immse(x_low./nor, x./nor);
mse_x_newton    = immse(x_newton./nor, x./nor);

psnr_x_low      = psnr(x_low./nor, x./nor);
psnr_x_newton   = psnr(x_newton./nor, x./nor);

ssim_x_low      = ssim(x_low./nor, x./nor);
ssim_x_newton	= ssim(x_newton./nor, x./nor);

%% DISPLAY
wndImg  = [0, 0.03];

figure(1); 
colormap(gray(256));

suptitle('Newton Method');
subplot(221);   imagesc(x,          wndImg);	axis image off;     title('ground truth');
subplot(222);   imagesc(x_full,     wndImg);  	axis image off;     title(['full-dose_{view : ', num2str(VIEW) '}']);
subplot(223);   imagesc(x_low,      wndImg);  	axis image off;     title({['low-dose_{view : ', num2str(VIEW) '}'], ['MSE : ' num2str(mse_x_low, '%.4e')], ['PSNR : ' num2str(psnr_x_low, '%.4f')], ['SSIM : ' num2str(ssim_x_low, '%.4f')]});
subplot(224);   imagesc(x_newton,   wndImg);  	axis image off;     title({['recon_{newton}'], ['MSE : ' num2str(mse_x_newton, '%.4e')], ['PSNR : ' num2str(psnr_x_newton, '%.4f')], ['SSIM : ' num2str(ssim_x_newton, '%.4f')]});

