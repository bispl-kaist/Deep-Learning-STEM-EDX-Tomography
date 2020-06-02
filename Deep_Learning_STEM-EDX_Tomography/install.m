clear;
clc;

%%
% Denosing CNN
network_path    = './Denoise_CNN/checkpoint/denoising/';

if ~isdir(network_path)
    mkdir(network_path);
end

ckpt_name       = [network_path 'checkpoint'];
ckpt_url        = 'https://www.dropbox.com/s/job1lmcbl4p57b7/checkpoint?dl=1';
index_name      = [network_path 'model.ckpt-30.index'];
index_url       = 'https://www.dropbox.com/s/6ygt4x3r04s0kkm/model.ckpt-30.index?dl=1';
network_name	= [network_path 'model.ckpt-30.data-00000-of-00001'];
network_url     = 'https://www.dropbox.com/s/2cjp1zsnly8s3fp/model.ckpt-30.data-00000-of-00001?dl=1';

fprintf('downloading Denoising CNN from %s\n', network_url) ;
websave(ckpt_name, ckpt_url);
websave(index_name, index_url);
websave(network_name, network_url);

% Super Resolution CNN
network_path    = './SR_CNN/checkpoint/sr/';

if ~isdir(network_path)
    mkdir(network_path);
end

ckpt_name       = [network_path 'checkpoint'];
ckpt_url        = 'https://www.dropbox.com/s/xbuwukgeor336is/checkpoint?dl=1';
index_name      = [network_path 'model.ckpt-1000.index'];
index_url       = 'https://www.dropbox.com/s/l5nd9z87k87gv00/model.ckpt-1000.index?dl=1';
network_name	= [network_path 'model.ckpt-1000.data-00000-of-00001'];
network_url     = 'https://www.dropbox.com/s/ai71xp3rjdqox5r/model.ckpt-1000.data-00000-of-00001?dl=1';

fprintf('downloading Super Resolution CNN from %s\n', network_url) ;
websave(ckpt_name, ckpt_url);
websave(index_name, index_url);
websave(network_name, network_url);
