clear ;

restoredefaultpath();
addpath('./lib_diff');

%%
N = 500;

x = single(rand(N, N, N));
b = single(rand(N, N, N));

n = int32(size(x));

if (length(n) < 3)
    n(3) = 1;
end

tic;
Ax = (Dx((x)));
Atb = (DxT((b)));
toc;

Ax_gpu = zeros(size(Ax), 'single');
Atb_gpu = zeros(size(Atb), 'single');

tic;
Dx_gpu(Ax_gpu, x, n);
DxT_gpu(Atb_gpu, b, n);
toc;

%%
Ax(:)'*b(:) - Atb(:)'*x(:)
Ax_gpu(:)'*b(:) - Atb(:)'*x(:)
