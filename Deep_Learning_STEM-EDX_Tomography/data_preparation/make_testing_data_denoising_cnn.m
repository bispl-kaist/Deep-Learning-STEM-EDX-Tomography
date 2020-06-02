clear; clc;

root_dir = './data/3D tomography/';
lst_dir = dir(root_dir);
lst_dir = lst_dir(3:end);

input_dir = '../Denoise_CNN/data/denoising/test/';

if ~isdir(input_dir)
    mkdir(input_dir)
end

for idir = 1:length(lst_dir)
    data_dir = lst_dir(idir).name;
    
    copyfile([root_dir data_dir './input_*.mat'], input_dir);
    
end
