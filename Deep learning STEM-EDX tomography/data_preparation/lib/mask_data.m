function [input, label] = mask_data(dataDir, saveDir, data, no_random, thres, init);

dataDir_label   = replace(dataDir, 'Result', 'Label');
saveDir_label   = replace(saveDir, 'Input', 'Label');

dataName        = 'SAIT';
lst_SAIT        = dir([dataDir_label '\' dataName '_*.mat']);

wgt             = 1.5;
ker_msk         = wgt*fspecial('average', 9);

for idx = 1: length(lst_SAIT)
    fileName    = lst_SAIT(idx).name;
    for idn = 1:no_random
        fileName_rec    = replace(fileName, data, [int2str(idn) '_' data]);       
        load([dataDir ,'\' ,fileName_rec], 'edx_img');
        init(:,:,idn)   =  edx_img; clear rec;
    end
    
    init            = init/max(init(:));
    edx_prev        = init; clear se_reg;
    
    load([dataDir_label, '\', fileName], 'edx_img');
    edx_img         = edx_img/max(edx_img(:));
    edx_label_prev  = edx_img; clear edx_img;
    
    edx_ker_label   = imfilter(edx_label_prev, ker_msk);
    edx_ker_label   = edx_ker_label/max(edx_ker_label(:));
    
    msk             = edx_ker_label > thres;
    
    edx_msk         = bsxfun(@times, edx_prev, msk);
    edx_img         = bsxfun(@times, edx_label_prev, msk);
    label           = edx_img;
    save([saveDir_label '\' fileName], 'edx_img');
    clear edx_img;
    
    edx_img     = edx_msk;
    input       = edx_img;
    save([saveDir '\' fileName], 'edx_img');
end