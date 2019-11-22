clear;

%%
nch         = 3;
npart       = 0;

for nepoch      = [50];

% nettype_set     = {'cgan', 'cgan_w_avgp'};
nettype_set     = {'cgan'};

% for navg    = 3:5;
for inettype = 1:length(nettype_set)
    
    nettype	= nettype_set{inettype};
    
for navg    = 3;
% for navg = [3:5, 6, 8, 10]
npatch      = 256;
nker        = 1;

% nid_set     = 1:8;
% nid_set     = 8:10;
% nid_set     = 11:12;

nid_set     = 1:12;

% nid_set     = [5, 8:12];
% nid_set     = [8, 12];
nid_offset  = 0;

% dir_haadf   = ['./../data/sait/avg' num2str(navg) '/test/'];
% dir_input   = ['./../data/sait/avg' num2str(navg) '/test/'];
dir_haadf   = ['./../data/sait/avg' num2str(3) '/test/'];
dir_input   = ['./../data/sait/avg' num2str(3) '/test/'];

% dir_output  = ['./../test/sait/' nettype '/avg' num2str(navg) '/epoch' num2str(nepoch) '/'];

dir_full_se     = ['./../test/sait/se/epoch' num2str(nepoch) '/avg' num2str(navg) '/'];
dir_full_s      = ['./../test/sait/s/epoch' num2str(nepoch) '/avg' num2str(navg) '/'];
dir_full_zn     = ['./../test/sait/zn/epoch' num2str(nepoch) '/avg' num2str(navg) '/'];


lst_data_se   	= dir(dir_full_se);
lst_data_s    	= dir(dir_full_s);
lst_data_zn   	= dir(dir_full_zn);


dir_data	= ['./data/sait_se_s_zn/' nettype '/epoch' num2str(nepoch) '/avg' num2str(navg) '/'];
% dir_data	= ['./data/sait/' nettype '/avg' num2str(navg) '/epoch' num2str(nepoch) '/'];

mkdir(dir_data);

disp(dir_data);

%%
nview       = 13;
nsize       = 256;
% nch         = 3;

wgt         = 1;

% for nid         = nid_set;    % 1, 2, 3, 4
for nid         = 5;    % 1, 2, 3, 4
% for nid         = 3;    % 1, 2, 3, 4
    %%
    sz_ker      = 11;
    std_ker     = fspecial('average', sz_ker);
    
    msk_ker     = fspecial('gaussian', 20, 3);
    
    [mx, my]    = meshgrid(linspace(-1, 1, nsize));
    
    switch (nid + nid_offset)
        case 1 % 'SAIT_012_1'
            %% 012_1
            circ_x_img_set	= [-10, 0, -5, -10, 0,  9, -6, -4, -2, 8, -7, -7, 0];
            circ_y_img_set  = [0, -15, 3, 0, 0,     1, 0, 0, 0, 0, 0, -3, 0];
            circ_msk_set    = [0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 0, 0];
            rot_img_set     = [4, 4, 4, 4, 5,       5, 5, 5, 5, 5, 5, 5, 5];
            
            msk             = single((mx/0.9).^2 + (my/2).^2 < 1);
            
        case 2 % 'SAIT_014_2'
            %% 014_2
            circ_x_img_set  = [-15, -2, 0, -2, -12,     -17, -3, -13, -20, -15,     -19, -12, -17];
            circ_y_img_set  = [0, 0, 0, 0, -2,           0, -2, 3, 5, -5,           0, 0, -3];
            circ_msk_set    = [0, 0, 0, 0, 0,           0, 0, 0, 0, 0,              0, 0, -15];
            rot_img_set 	= [5, 6, 7, 7, 10,          10, 10, 11, 10, 12,         13, 11, 10];
            
            msk             = single((mx/0.5).^2 + (my/2).^2 < 1);
            
        case 3 % 'SAIT_015_1'
            %% 015_1
            circ_x_img_set  = [15, -6, 8, 0, 0,     2, 5, 2, -5, -7,    5, -15, 4];
            circ_y_img_set	= [0, 0, 0, 2, 0,       3, 2, -5, 0, -3,     0, 0, -5];
            circ_msk_set    = [0, 0, 0, 0, 0,       0, 0, 0, 0, 0,      0, 0, 0];
            rot_img_set     = [0, 0, 0, 0, 1,       2, 2, 2, 2, 2,      3, 3, 3];
            
            msk             = single((mx/0.3).^2 + (my/2).^2 < 1);
            
        case 4 % 'SAIT_015_2'
            %% 015_2
            circ_x_img_set  = [-15, -13, 5, -8, -20,    -5, -8, -5, 0, 10,   -5, 15, -10];
            circ_y_img_set	= [0, -1, -3, -2, 1,        -6, -6, -5, 0, 0,    -3, 2, -10];
            circ_msk_set    = [0, 0, 0, 0, 0,           0, 0, 0, 0, 0,      0, 0, 0];
            rot_img_set   	= [0, 0, 0, 0, 0,           0, 0, -3, 0, 0,      0, 0, 0];
            
            msk             = single(((mx+0.15)/0.6).^2 + (my/1).^2 < 1);
            
        case 5 % 'SAIT_026_1'
            %% 025_1
            circ_x_img_set  = [10, 8, -2, 10, 15,   -3, 0, -1, 13, 0,        12, 0, 12];
            circ_y_img_set	= [0, -7, 1, 5, 7,      3, 2, 4, 3, 2,          0, 0, 5];
            circ_msk_set    = [0, 0, 0, 0, 0,       0, 0, 0, 0, 0,          0, 0, 0];
            rot_img_set   	= [0, -2, -3, -3, -4,   -4, -4, -4, -5, -3,     -3, -3, -1];
            
            msk             = single((mx/0.4).^2 + (my/2).^2 < 1);
            
        case 6 % 'SAIT_025_1'
            %% 025_1
            circ_x_img_set  = [5, 10, 22, 5, 15,        20, 5, 10, 0, 0,       0, -5, 0];
            circ_y_img_set	= [10, -5, 5, -15, -10,     -20, -10, -5, -5, 0,       5, 5, 0];
            circ_msk_set    = [0, 0, 0, 0, 0,           0, 0, 0, 0, 0,          0, 0, 0];
            rot_img_set   	= [0, 5, 8, 12, 15,         15, 15, 20, 15, 15,     15, 15, 15];
            
            msk             = single((mx/0.4).^2 + (my/0.8).^2 < 1);
            
        case 7 % 'SAIT_025_2'
            %% 025_2
            circ_x_img_set  = [0, 0, 0, 0, 0, 0, -10, 0, 0, -10, -5, -10, -5];
            circ_y_img_set	= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10, -5, 0];
            circ_msk_set    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            rot_img_set   	= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            
            msk             = single((mx/0.5).^2 + (my/0.5).^2 < 1);
            
%         case 8 % 'SAIT_025_3'
%             %% 025_3
%             circ_x_img_set  = [0, 3, 0, 5, -5,      -5, -10, -20, -18, -30,   -30, -45, -55]+35;
%             circ_y_img_set	= [0, 8, 10, 0, 0,      8, 13, 18, 18, 23,      20, 23, 25]+20;
%             circ_msk_set    = [0, 0, 0, 0, 0,       0, 0, 0, 0, 0,          0, 0, 0];
%             rot_img_set   	= [0, 3, 5, 8, 10,      10, 10, 10, 12, 10,     5, 5, 3]+15;
%             
%             msk             = single(((mx - 0.5)/1).^2 + ((my + 0.1)/1.5).^2 < 1);
%             msk(1:70, :)    = 0;

        case 8 % 'SAIT_028_1'
            %% 028_1
            circ_x_img_set  = [10, 0, 0, 0, -5,     0, -5, -5, 5, 3,        0, 3, 0];
            circ_y_img_set	= [27, 10, 11, 10, 17,	25, 15, 20, 18, 14,     15, 15, 15];
            circ_msk_set    = [0, 0, 0, 0, 0,       0, 0, 0, 0, 0,          0, 0, 0];
            rot_img_set   	= [8, 8, 9, 10, 10,     10, 10, 10, 10, 10,     7, 5, 5];
            
            msk             = single((mx/0.7).^2 + (my/4).^2 < 1);
%             msk(1:70, :)    = 0;
            
        case 9 % 'SAIT_028_2'
            %% 028_2
            circ_x_img_set  = [-5, -5, -2, -0, 3,       -3, 0, -3, -3, -3,          0, 0, -0];
            circ_y_img_set	= [0, -2, 5, 2, 0,          2, 4, 2, 2, 5,              8, 3, 3];
            circ_msk_set    = [0, 0, 0, 0, 0,           0, 0, 0, 0, 0,              0, 0, 0];
            rot_img_set   	= [-5, -7, -9, -12, -14,	-18, -18, -18, -18, -18,  	-15, -10 -8];
            
            msk             = single((mx/0.7).^2 + (my/4).^2 < 1);
%             msk(1:70, :)    = 0;
            
        case 10 % 'SAIT_028_3'
            %% 028_3
            circ_x_img_set  = [0, 5, 0, -3, 2,	0, -10, 0, -8, 15,  -5, 0, -15];
            circ_y_img_set	= [0, 0, 3, 5, -5,	8, 0, 13, 15, 15,  	15, 10, 5];
            circ_msk_set    = [0, 0, 0, 0, 0,	0, 0, 0, 0, 0,      0, 0, 0];
            rot_img_set   	= [0, 0, 0, 0, 0,	0, 0, 0, 0, 0,      0, 0, 0];
            
            msk             = single((mx/0.7).^2 + (my/4).^2 < 1);
%             msk(1:70, :)    = 0;

        case 11 % 'SAIT_028_4'
            %% 028_4
            circ_x_img_set  = [0, 8, -3, 5, -5,     10, -5, 0, 10, 10,	10, -5, 5];
            circ_y_img_set	= [3, 0, 0, 0, 0,       0, 5, -7, -8, -5,   -5, 0, 0];
            circ_msk_set    = [0, 0, 0, 0, 0,       0, 0, 0, 0, 0,      0, 0, 0];
            rot_img_set   	= [0, 0, 0, 0, 0,       0, 0, 0, 0, 0,      0, 0, 0];
            
            msk             = ones(256, 256, 13, 'single');
            
            msk(:,:,1)       = 1 - single(((mx + 1.1)/1).^2 + ((my - 1.1)/1).^2 < 1);
            msk(:,:,2)       = 1 - single(((mx + 1.1)/1).^2 + ((my - 1.1)/1).^2 < 1);
%             msk             = single(((mx)/4).^2 + (my/4).^2 < 1);
%             msk             = 1;
            msk(215:256, :,:)	= 0;

        case 12 % 'SAIT_028_5'
            %% 028_5
            circ_x_img_set  = [-15, 0, -15, -10, -5,         5, 0, 0, 5, 0,      5, 0, 0];
            circ_y_img_set	= [-15, -20, -20, -20, -25,     -28, -20, -25, -27, -27,      -27, -27, -27];
            circ_msk_set    = [0, 0, 0, 0, 0,	0, 0, 0, 0, 0,      0, 0, 0];
            rot_img_set   	= [0, 0, 0, 0, 0,	0, 0, 0, 0, 0,      0, 0, 0];
            
            msk             = ones(256, 256, 13, 'single');
            
            msk_            = single(((mx + 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1) + single(((mx + 1.2)/1).^2 + ((my - 0.8)/1).^2 < 1);
            msk_            = ~single(msk_ > 0);
            msk(:,:,1)       = msk_;
            
            msk_            = single(((mx + 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1) + single(((mx + 0.5)/1).^2 + ((my - 1.41)/1).^2 < 1);
            msk_            = ~single(msk_ > 0);
            msk(:,:,2)       = msk_;
            
            msk_            = single(((mx - 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1) + single(((mx + 0.5)/1).^2 + ((my - 1.41)/1).^2 < 1);
            msk_            = ~single(msk_ > 0);
            msk(:,:,3)       = msk_;
            
            
            msk_            = single(((mx - 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1) + single(((mx - 0.6)/1).^2 + ((my - 1.40)/1).^2 < 1);
            msk_            = ~single(msk_ > 0);
            msk(:,:,5)       = msk_;
            
            
            msk_            = single(((mx - 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1) + single(((mx - 0.6)/1).^2 + ((my - 1.40)/1).^2 < 1);
            msk_            = ~single(msk_ > 0);
            msk(:,:,6)   = msk_;
            msk(:,:,7)   = msk_;
            msk(:,:,8)   = msk_;
            msk(:,:,9)   = msk_;
            msk(:,:,10)   = msk_;
            msk(:,:,11)   = msk_;
            msk(:,:,12)   = msk_;
            msk(:,:,13)   = msk_;
% %             
% %             msk_            = single(((mx - 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1) + single(((mx - 0.6)/1).^2 + ((my - 1.40)/1).^2 < 1);
% %             msk_            = ~single(msk_ > 0);
% %             msk(:,:,7)       = msk_;
% %             
% %             msk_            = single(((mx - 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1) + single(((mx - 0.6)/1).^2 + ((my - 1.40)/1).^2 < 1);
% %             msk_            = ~single(msk_ > 0);
% %             msk(:,:,8)       = msk_;
% %             
% %             
% %             msk_            = single(((mx - 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1) + single(((mx - 0.6)/1).^2 + ((my - 1.40)/1).^2 < 1);
% %             msk_            = ~single(msk_ > 0);
% %             msk(:,:,8)       = msk_;
%             
%             msk(:,:,11)      = 1 - single(((mx + 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1);
%             msk(:,:,12)      = 1 - single(((mx + 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1);
%             msk(:,:,13)      = 1 - single(((mx + 1.45)/1).^2 + ((my + 0.0)/1).^2 < 1);
            
%             msk             = single((mx/4).^2 + (my/4).^2 < 1);
            msk(200:256, :)    = 0;
            msk(:,1:50,11:13)  = 0;
    end
    
    msk     = imfilter(msk, msk_ker);
    
    %%
    haadfs          = zeros([nsize, nsize, 1, nview], 'single');
    haadfs_w_msk	= zeros([nsize, nsize, 1, nview], 'single');
    inputs      = zeros([nsize, nsize, nch, nview], 'single');
    outputs_std	= zeros([nsize, nsize, nch, nview], 'single');
    outputs_cnn	= zeros([nsize, nsize, nch, nview], 'single');
    
    th          = 0.0;
    
    
    %%
    
    figure(1);
    set(gca,'position',[0 0 1 1]);
    set(gcf,'PaperPositionMode','auto');
    
    
    figure(2);
    set(gca,'position',[0 0 1 1]);
    set(gcf,'PaperPositionMode','auto');
    
    
    figure(3);
    set(gca,'position',[0 0 1 1]);
    set(gcf,'PaperPositionMode','auto');
    
    %%
    for iview = 1:nview
        %    fid      = fopen([dataDir inputLst((nid - 1)*nview + iview).name], 'rb');
        %     fid         = fopen([inputDir num2str((nid - 1)*nview + iview - 1) '-raw_inputs.raw'], 'rb');
        %     input       = permute(reshape(fread(fid, 'single'), [nch, nsize, nsize]), [3, 2, 1]);
        %     fclose(fid);
        %     input(input < 0)     = 0;
        
        %    fid      = fopen([dataDir outputLst((nid - 1)*nview + iview).name], 'rb');
        %     fid         = fopen([inputDir num2str((nid - 1)*nview + iview - 1) '-raw_outputs.raw'], 'rb');
        %     output_cnn  = permute(reshape(fread(fid, 'single'), [nch, nsize, nsize]), [3, 2, 1]);
        %     fclose(fid);
        %     output_cnn(output_cnn < 0)	= 0;
        
        load([dir_haadf 'haadf_' num2str((nid - 1)*nview + iview - 1, '%04d') '.mat']);
        load([dir_input 'input_' num2str((nid - 1)*nview + iview - 1, '%04d') '.mat']);
        
%         load([dir_output 'output_' num2str((nid - 1)*nview + iview - 1, '%04d') '.mat']);
        

        load([dir_full_se 'output_' num2str((nid - 1)*nview + iview - 1, '%04d') '.mat']); 	output_se   = output;
        load([dir_full_s  'output_' num2str((nid - 1)*nview + iview - 1, '%04d') '.mat']); 	output_s    = output;
        load([dir_full_zn 'output_' num2str((nid - 1)*nview + iview - 1, '%04d') '.mat']);	output_zn   = output;

        output(:,:,1)   = output_se;
        output(:,:,2) 	= output_s;
        output(:,:,3)   = output_zn;

        output_cnn  = output;
        
        %
        output_std	= imfilter(input, std_ker, 'replicate', 'same')*sqrt(11);
        output_std(output_std < th)	= 0;
        
        %%
        if (nid + nid_offset) == 11 || (nid + nid_offset) == 12
            msk_    	= msk(:,:,iview);
        else
            msk_        = circshift(msk, circ_msk_set(iview), 2);
        end
%         msk_        = 1;

%         haadf       = imrotate(haadf, rot_img_set(iview), 'bilinear','crop');
%         haadf       = circshift(haadf, circ_x_img_set(iview), 2);
%         haadf       = circshift(haadf, circ_y_img_set(iview), 1);
%         haadf_w_msk	= bsxfun(@times, haadf, msk_);
        
%         input       = imrotate(input, rot_img_set(iview), 'bilinear','crop');
%         input       = circshift(input, circ_x_img_set(iview), 2);
%         input       = circshift(input, circ_y_img_set(iview), 1);
        %     input       = bsxfun(@times, input, msk_);
        
%         output_cnn  = imrotate(output_cnn, rot_img_set(iview), 'bilinear','crop');
%         output_cnn  = circshift(output_cnn, circ_x_img_set(iview), 2);
%         output_cnn  = circshift(output_cnn, circ_y_img_set(iview), 1);
%         output_cnn 	= bsxfun(@times, output_cnn, msk_);
        
%         output_std	= imrotate(output_std, rot_img_set(iview), 'bilinear','crop');
%         output_std  = circshift(output_std, circ_x_img_set(iview), 2);
%         output_std  = circshift(output_std, circ_y_img_set(iview), 1);
%         output_std 	= bsxfun(@times, output_std, msk_);
        
        output_cnn(output_cnn < th) = 0;
        %%
%         haadfs(:,:,:,iview)         = haadf;
%         haadfs_w_msk(:,:,:,iview)	= haadf_w_msk;
        
        inputs(:,:,:,iview)         = input;
        outputs_cnn(:,:,:,iview)	= output_cnn;
        outputs_std(:,:,:,iview)	= output_std;
        
        input_se                    = input(:,:,1);
        input_s                     = input(:,:,2);
        input_zn                    = input(:,:,3);
        
        output_std_se               = output_std(:,:,1);   
        output_std_s                = output_std(:,:,2);
        output_std_zn               = output_std(:,:,3);
        
        output_cnn_se               = output_cnn(:,:,1);
        output_cnn_s                = output_cnn(:,:,2);
        output_cnn_zn               = output_cnn(:,:,3);
        
        
%         output_cnn(:, 128,:)        = 1;
%         output_cnn(128,:, :)        = 1;

disp(iview)

figure(1); imagesc(input);
% figure(2); imagesc(output_std);
figure(3); imagesc(output_cnn);

pause();
        
%         figure(navg);
%         subplot(341); imagesc(input);       title(nid);
%         subplot(342); imagesc(input_se);
%         subplot(343); imagesc(input_s);
%         subplot(344); imagesc(input_zn);
%         
%         subplot(345); imagesc(output_std);
%         subplot(346); imagesc(output_std_se);
%         subplot(347); imagesc(output_std_s);
%         subplot(348); imagesc(output_std_zn);
% 
% %         subplot(345); imagesc(haadf);
% %         subplot(346); imagesc(haadf_w_msk);
% %         subplot(347); imagesc(output_std_s);
% %         subplot(348); imagesc(output_std_zn);
% 
%         
%         subplot(349); imagesc(output_cnn);  title(iview);
%         subplot(3,4,10); imagesc(output_cnn_se);
%         subplot(3,4,11); imagesc(output_cnn_s);
%         subplot(3,4,12); imagesc(output_cnn_zn);
%         
% %         pause();
%         drawnow();
        
    end
    
    %%
%     haadfs          = squeeze(haadfs);
%     haadfs_w_msk 	= squeeze(haadfs_w_msk);
%     
%     se_inputs       = squeeze(inputs(:,:,1,:));
%     s_inputs        = squeeze(inputs(:,:,2,:));
%     zn_inputs       = squeeze(inputs(:,:,3,:));
%     
%     se_outputs_std	= squeeze(outputs_std(:,:,1,:));
%     s_outputs_std 	= squeeze(outputs_std(:,:,2,:));
%     zn_outputs_std  = squeeze(outputs_std(:,:,3,:));
%     
%     se_outputs_cnn	= squeeze(outputs_cnn(:,:,1,:));
%     s_outputs_cnn 	= squeeze(outputs_cnn(:,:,2,:));
%     zn_outputs_cnn  = squeeze(outputs_cnn(:,:,3,:));
%     
%     %%
%     save([dir_data 'imdb_sait_tomo' num2str(nid + nid_offset) '.mat'],  ...
%         'haadfs',           'haadfs_w_msk',                           	...
%         'se_inputs',        's_inputs',         'zn_inputs',            ...
%         'se_outputs_std',   's_outputs_std',    'zn_outputs_std',       ...
%         'se_outputs_cnn',   's_outputs_cnn',    'zn_outputs_cnn');
end
end
end
end