clear; clc;

dataDir_se  = 'attention_data\1st_atom\Result\SAIT\'; % the directory of result of attention network for se
dataDir_s   = 'attention_data\2nd_atom\Result\SAIT\'; % the directory of result of attention network for s
dataDir_zn  = 'attention_data\3rd_atom\Result\SAIT\'; % the directory of result of attention network for zn
dataName    = 'SAIT';

saveDir     = 'result\'; % the directory of final result [se, s, zn]

lst_SAIT    = dir([dataDir_se '\' dataName '_*.mat']);

for ilst = 1:length(lst_SAIT)
    load([dataDir_se '\' lst_SAIT(ilst).name], 'rec');
    se_data     = rec;
    load([dataDir_s '\' replace(lst_SAIT(ilst).name, 'sei', 'si')], 'rec');
    s_data      = rec;
    load([dataDir_zn '\' replace(lst_SAIT(ilst).name, 'sei', 'zni')], 'rec');
    zn_data     = rec;
    rec_img     = cat(3, se_data, s_data, zn_data);
    save([saveDir, lst_SAIT(ilst).name(1:end-7)], 'rec_img');
end