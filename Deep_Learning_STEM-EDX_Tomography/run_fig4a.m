clear ;

%% 
data_dir = './result/proposed/';

%%
load([data_dir 'recon_0000_se.mat']);
nor_x = 1.2*mean(max(max(x, [], 1), [], 2));
recon_se = bsxfun(@times, single(x), 1./nor_x);

load([data_dir 'recon_0000_s.mat']);
nor_x = 1.2*mean(max(max(x, [], 1), [], 2));
recon_s = bsxfun(@times, single(x), 1./nor_x);

load([data_dir 'recon_0000_zn.mat']);
nor_x = 1.2*mean(max(max(x, [], 1), [], 2));
recon_zn = bsxfun(@times, single(x), 1./nor_x);

%%
wnd = [0.1, 1.1];
recon = cat(4, recon_se, recon_s, recon_zn) ./ wnd(2);
recon(recon < wnd(1)) = 0;

clear recon_se recon_s recon_zn x;

%% 
dir = 3;

switch dir
    case 1
        recon_dir	= rot90(permute(recon, [1, 2, 4, 3]), 1);
    case 2
        recon_dir	= flip(permute(recon, [3, 2, 4, 1]), 1);
    case 3
        recon_dir	= flip(permute(recon, [3, 1, 4, 2]), 1);
end

bnd = 1:192;
ptc = 1:80;

iy = 45;
ix = 91;

islice = 115;

recon_edx = recon_dir(:,:,:,islice);
recon_se = zeros(size(recon_edx));      recon_se(:,:,1) = recon_dir(:,:,1,islice);
recon_s = zeros(size(recon_edx));       recon_s(:,:,2) = recon_dir(:,:,2,islice);
recon_zn = zeros(size(recon_edx));      recon_zn(:,:,3) = recon_dir(:,:,3,islice);

%%
figure('name', 'Figure. 4(a)'); 
subplot(3,2,[1,5]); imagesc(recon_edx(:, bnd + 32, :));         axis image; axis off;   title('Figure. 4(a)');
subplot(3,2,2);     imagesc(recon_se(ptc + iy, ptc + ix, :));   axis image; axis off;   title('se');
subplot(3,2,4);     imagesc(recon_s(ptc + iy, ptc + ix, :));    axis image; axis off;   title('s');
subplot(3,2,6);     imagesc(recon_zn(ptc + iy, ptc + ix, :));   axis image; axis off;   title('zn');
