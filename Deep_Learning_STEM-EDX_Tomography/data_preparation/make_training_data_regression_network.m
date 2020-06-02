clear; clc;
addpath('lib');
%% BLURRING KERNEL

sz_ker      = 11;
ker         = fspecial('average', sz_ker);

nor_stem    = 255.0;
nor_edx     = 1.0;
wnd         = [0, 1];

dataDir     = 'data\CNN-based kernel';
dataName    = 'SAIT';

lst_SAIT    = dir([dataDir '\' dataName '_*.mat']);

for ilst = 1:length(lst_SAIT)
    load([dataDir '\' lst_SAIT(ilst).name]);
    
    haadf_stem              = single(haadf_stem)./nor_stem;
    Sei_img                 = single(Sei_img)./nor_edx;
    Si_img                  = single(Si_img)./nor_edx;
    Zni_img                 = single(Zni_img)./nor_edx;
    
    [haadf_stem_reg, param]	= registration(haadf_stem, haadf_stem);
    Sei_img_reg             = registration(Sei_img, haadf_stem, param);
    Si_img_reg              = registration(Si_img, haadf_stem, param);
    Zni_img_reg             = registration(Zni_img, haadf_stem, param);
    
    %% label data
    edx_img                 = mean(Sei_img_reg, 3);
    save(['regression_data\1st_atom\Label\' replace(lst_SAIT(ilst).name, '.mat', '_sei.mat')], 'edx_img' , '-v6'); % the directory of label of regression network for se
    clear edx_img;
    
    edx_img                 = mean(Si_img_reg, 3);
    save(['regression_data\2nd_atom\Label\' replace(lst_SAIT(ilst).name, '.mat', '_si.mat')], 'edx_img' , '-v6'); % the directory of label of regression network for s
    clear edx_img;
    
    edx_img                 = mean(Zni_img_reg, 3);
    save(['regression_data\3rd_atom\Label\' replace(lst_SAIT(ilst).name, '.mat', '_zni.mat')], 'edx_img' , '-v6'); % the directory of label of regression network for zn
    clear edx_img;
    
    %% bootstrap subsampled data
    ind_set         = 1:size(Zni_img_reg,3);
    for irand = 1:min(30, nchoosek(size(Zni_img_reg,3),8)) % The number of bootstrap subsampling is set to 30.
        ind         = randperm(length(ind_set), 8);
        il
        edx_img     = mean(Sei_img_reg(:,:,ind_set(ind)), 3);
        save(['regression_data\1st_atom\Input\' replace(lst_SAIT(ilst).name, '.mat', [ '_' int2str(irand) '_sei.mat'])], 'edx_img' , '-v6'); % the directory of label of regression network for se
        sei_img     = edx_img;
        clear edx_img;
        
        edx_img     = mean(Si_img_reg(:,:,ind_set(ind)), 3);
        save(['regression_data\2nd_atom\Input\' replace(lst_SAIT(ilst).name, '.mat', [ '_' int2str(irand) '_si.mat'])], 'edx_img' , '-v6');
        clear edx_img;
        
        edx_img     = mean(Zni_img_reg(:,:,ind_set(ind)), 3);
        save(['regression_data\3rd_atom\Input\' replace(lst_SAIT(ilst).name, '.mat', [ '_' int2str(irand) '_zni.mat'])], 'edx_img' , '-v6');
        
        edx_img     = cat(3, sei_img, edx_img); % concatenation of se and zn for training of s
        save(['regression_data\2nd_atom\Input_concat\' lst_SAIT(ilst).name(1:end-4) '_' int2str(irand) '_si.mat'], 'edx_img' , '-v6');  
        clear edx_img;
    end
    
end
