clear;

%%
netType     = 'w_dropout';  % 'w_dropout', 'wo_dropout'
avgType     = 'avg3';       % 'avg1', 'avg2', 'avg3', 'avg4'
trnType     = 'each';        % 'all', 'core', 'shell'

outputDir   = ['./data/' netType '/' avgType '/' trnType '/'];
inputDir_se = ['./../sait_tomo_test/' netType '/' avgType '/' trnType '/se/raws/'];
inputDir_s  = ['./../sait_tomo_test/' netType '/' avgType '/' trnType '/s/raws/'];
inputDir_zn	= ['./../sait_tomo_test/' netType '/' avgType '/' trnType '/zn/raws/'];

inputDirLst_se	= dir([inputDir_se '*-raw_inputs.raw']);
inputDirLst_s	= dir([inputDir_s '*-raw_inputs.raw']);
inputDirLst_zn	= dir([inputDir_zn '*-raw_inputs.raw']);
outputDirLst_se	= dir([inputDir_se '*-raw_outputs.raw']);
outputDirLst_s	= dir([inputDir_s '*-raw_outputs.raw']);
outputDirLst_zn	= dir([inputDir_zn '*-raw_outputs.raw']);

mkdir(outputDir);

%%
nview       = 13;
nsize       = 256;

if strcmp(trnType, 'all')
    nch     = 3;
else
    nch     = 1;
end

for nid         = 1:4;    % 1, 2, 3, 4

%%
sz_ker      = 11;
std_ker     = fspecial('average', sz_ker);

msk_ker     = fspecial('gaussian', 20, 3);

[mx, my]    = meshgrid(linspace(-1, 1, nsize));

switch nid
    case 1 % 'SAIT_012_1'
        %% 012_1
        circ_x_img_set	= [-10, 0, -5, -10, 0, 6, -5, -4, -2, 8, -7, -5, 0];
        circ_y_img_set  = [0, -15, 5, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0];
        circ_msk_set    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        rot_img_set     = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        
        msk             = single((mx/0.9).^2 + (my/2).^2 < 1);
        
    case 2 % 'SAIT_014_2'
        %% 014_2
        circ_x_img_set  = [-15, 0, 3, 0, -10, -15, 0, -10, -17, -10, -14, -8, -15];
        circ_y_img_set  = [0, 0, 0, 0, 0, 2, 0, 3, 5, -5, 0, 0, -3];
        circ_msk_set    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15];
        rot_img_set 	= [5, 5, 7, 7, 10, 10, 10, 11, 10, 10, 10, 7, 7];
        
        msk             = single((mx/0.5).^2 + (my/2).^2 < 1);
        
    case 3 % 'SAIT_015_1'
        %% 015_1
        circ_x_img_set  = [15, -5, 8, 0, 0, 0, 5, 0, -5, -5, 5, -15, 2];
        circ_y_img_set	= [0, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, -5];
        circ_msk_set    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        rot_img_set     = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        
        msk             = single((mx/0.5).^2 + (my/2).^2 < 1);
        
    case 4 % 'SAIT_015_2'
        %% 015_2
        circ_x_img_set  = [0, 0, 15, 8, 0, 20, 20, 20, 20, 30, 15, 40, 10];
        circ_y_img_set	= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        circ_msk_set    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        rot_img_set   	= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        
        msk             = single((mx/0.6).^2 + (my/2).^2 < 1);
        
end
msk     = imfilter(msk, msk_ker);

%%
inputs      = zeros([nsize, nsize, 3, nview], 'single');
outputs_std	= zeros([nsize, nsize, 3, nview], 'single');
outputs_cnn	= zeros([nsize, nsize, 3, nview], 'single');

%%
for iview = 1:nview
    %    fid      = fopen([dataDir inputLst((nid - 1)*nview + iview).name], 'rb');
    fid         = fopen([inputDir_se num2str((nid - 1)*nview + iview - 1) '-raw_inputs.raw'], 'rb');
    input_se    = permute(reshape(fread(fid, 'single'), [nch, nsize, nsize]), [3, 2, 1]);
    fclose(fid);
    input_se(input_se < 0)     = 0;
    
    fid         = fopen([inputDir_s num2str((nid - 1)*nview + iview - 1) '-raw_inputs.raw'], 'rb');
    input_s     = permute(reshape(fread(fid, 'single'), [nch, nsize, nsize]), [3, 2, 1]);
    fclose(fid);
    input_s(input_s < 0)     = 0;
    
    fid         = fopen([inputDir_zn num2str((nid - 1)*nview + iview - 1) '-raw_inputs.raw'], 'rb');
    input_zn    = permute(reshape(fread(fid, 'single'), [nch, nsize, nsize]), [3, 2, 1]);
    fclose(fid);
    input_zn(input_zn < 0)     = 0;
        
    
    %    fid      = fopen([dataDir outputLst((nid - 1)*nview + iview).name], 'rb');
    fid             = fopen([inputDir_se num2str((nid - 1)*nview + iview - 1) '-raw_outputs.raw'], 'rb');
    output_cnn_se   = permute(reshape(fread(fid, 'single'), [nch, nsize, nsize]), [3, 2, 1]);
    fclose(fid);
    output_cnn_se(output_cnn_se < 0)     = 0;
    
    fid             = fopen([inputDir_s num2str((nid - 1)*nview + iview - 1) '-raw_outputs.raw'], 'rb');
    output_cnn_s    = permute(reshape(fread(fid, 'single'), [nch, nsize, nsize]), [3, 2, 1]);
    fclose(fid);
    output_cnn_s(output_cnn_s < 0)     = 0;
    
    fid             = fopen([inputDir_zn num2str((nid - 1)*nview + iview - 1) '-raw_outputs.raw'], 'rb');
    output_cnn_zn   = permute(reshape(fread(fid, 'single'), [nch, nsize, nsize]), [3, 2, 1]);
    fclose(fid);
    output_cnn_zn(output_cnn_zn < 0)     = 0;
    
    %
    input       = cat(3, input_se, input_s, input_zn);
    output_cnn	= cat(3, output_cnn_se, output_cnn_s, output_cnn_zn);
    output_std	= imfilter(input, std_ker, 'replicate', 'same')*sqrt(11);
    output_std(output_std < 0)	= 0;
    
    %%
    msk_        = circshift(msk, circ_msk_set(iview), 2);
    
    input       = imrotate(input, rot_img_set(iview), 'bilinear','crop');
    input       = circshift(input, circ_x_img_set(iview), 2);
    input       = circshift(input, circ_y_img_set(iview), 1);
    %     input       = bsxfun(@times, input, msk_);
    
    output_cnn  = imrotate(output_cnn, rot_img_set(iview), 'bilinear','crop');
    output_cnn  = circshift(output_cnn, circ_x_img_set(iview), 2);
    output_cnn  = circshift(output_cnn, circ_y_img_set(iview), 1);
    output_cnn 	= bsxfun(@times, output_cnn, msk_);
    
    output_std	= imrotate(output_std, rot_img_set(iview), 'bilinear','crop');
    output_std  = circshift(output_std, circ_x_img_set(iview), 2);
    output_std  = circshift(output_std, circ_y_img_set(iview), 1);
    output_std 	= bsxfun(@times, output_std, msk_);
    
    %%
    %     iy                      = 115;
    %     ix                      = 128;
    %
    %     input(:, ix, :)       	= 1;
    %     input(iy, :, :)         = 1;
    %
    %     output_cnn(:, ix, :)	= 1;
    %     output_cnn(iy, :, :)	= 1;
    %
    %     output_std(:, ix, :)	= 1;
    %     output_std(iy, :, :)	= 1;
    
    %%
    inputs(:,:,:,iview)         = input;
    outputs_cnn(:,:,:,iview)	= output_cnn;
    outputs_std(:,:,:,iview)	= output_std;
    
    figure(10);
    subplot(131); imagesc(input);
    subplot(132); imagesc(output_std);
    subplot(133); imagesc(output_cnn);
    
    pause();
%     drawnow();
    
end

%%
se_inputs       = squeeze(inputs(:,:,1,:));
s_inputs        = squeeze(inputs(:,:,2,:));
zn_inputs       = squeeze(inputs(:,:,3,:));

se_outputs_std	= squeeze(outputs_std(:,:,1,:));
s_outputs_std 	= squeeze(outputs_std(:,:,2,:));
zn_outputs_std  = squeeze(outputs_std(:,:,3,:));

se_outputs_cnn	= squeeze(outputs_cnn(:,:,1,:));
s_outputs_cnn 	= squeeze(outputs_cnn(:,:,2,:));
zn_outputs_cnn  = squeeze(outputs_cnn(:,:,3,:));

%%
save([outputDir 'imdb_sait_tomo' num2str(nid) '.mat'], ...
    'se_inputs',        's_inputs',         'zn_inputs',        ...
    'se_outputs_std',   's_outputs_std',    'zn_outputs_std',   ...
    'se_outputs_cnn',   's_outputs_cnn',    'zn_outputs_cnn');
end