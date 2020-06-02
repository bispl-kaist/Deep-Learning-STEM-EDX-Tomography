%% SQD2 Analysis
load('s_proposed_rec256_1.mat')
s_recon = x1;
load('se_proposed_rec256_1.mat')
se_recon = x1;
load('zn_proposed_rec256_1.mat')
zn_recon = x1;

Particle_s_1 = s_recon(:,:,1:100);
Particle_se_1 = se_recon(:,:,1:100);
Particle_zn_1 = zn_recon(:,:,1:100);

clear x1 s_recon se_recon zn_recon;

load('s_proposed_rec256_2.mat')
s_recon = x2;
load('se_proposed_rec256_2.mat')
se_recon = x2;
load('zn_proposed_rec256_2.mat')
zn_recon = x2;

Particle_s_2 = s_recon(:,:,60:150);
Particle_se_2 = se_recon(:,:,60:150);
Particle_zn_2 = zn_recon(:,:,60:150);

clear x2 s_recon se_recon zn_recon;

Particle_s = Particle_s_1;          % Choose 1 for SQD2-1 or 2 for SQD2-2
Particle_se = Particle_se_1;        % Choose 1 for SQD2-1 or 2 for SQD2-2
Particle_zn = Particle_zn_1;        % Choose 1 for SQD2-1 or 2 for SQD2-2

x_size = size(Particle_s,1);
y_size = size(Particle_s,2);
z_size = size(Particle_s,3);

Particle_s_b = zeros(256,256,128);
Particle_s_b(:,:,1:z_size) = Particle_s;
Particle_se_b = zeros(256,256,128);
Particle_se_b(:,:,1:z_size) = Particle_se;
Particle_zn_b = zeros(256,256,128);
Particle_zn_b(:,:,1:z_size) = Particle_zn;

x_size = size(Particle_s_b,1);
y_size = size(Particle_s_b,2);
z_size = size(Particle_s_b,3);

[xx yy zz] = meshgrid(-x_size/2:x_size/2-1,-y_size/2:y_size/2-1,-z_size/2:z_size/2-1);
[AZ, EL, R] = cart2sph(xx,yy,zz);
AZ = rad2deg(AZ)+180;
EL = rad2deg(EL);
clear xx yy zz;

Particle_s_mask = Particle_s_b > 0.003;
Particle_s_thr = Particle_s_mask .* Particle_s_b;
Particle_se_mask = Particle_se_b > 0.003;
Particle_se_thr = Particle_se_mask .* Particle_se_b;
Particle_zn_mask = Particle_zn_b > 0.003;
Particle_zn_thr = Particle_zn_mask .* Particle_zn_b;

com = centerOfMass(Particle_se_thr);
com_x = int16(com(1) - x_size/2);
com_y = int16(com(2) - y_size/2);
com_z = int16(com(3) - z_size/2);

Particle_s_thr = circshift(Particle_s_thr,[-com_x -com_y -com_z]);
Particle_se_thr = circshift(Particle_se_thr,[-com_x -com_y -com_z]);
Particle_zn_thr = circshift(Particle_zn_thr,[-com_x -com_y -com_z]);

Particle_s_mask = circshift(Particle_s_mask,[-com_x -com_y -com_z]);
Particle_se_mask = circshift(Particle_se_mask,[-com_x -com_y -com_z]);
Particle_zn_mask = circshift(Particle_zn_mask,[-com_x -com_y -com_z]);

mm = 1;
tmp = ((AZ >= 45) & (AZ < 45+1));
ttmp = ((EL >= 0) & (EL < 0+1));
tttmp = tmp & ttmp;
IDX = find(tttmp == 1);

thickness_s = zeros(181,360);
thickness_se = zeros(181,360);
thickness_zn = zeros(181,360);

for ii=0:1:359
    nn = 1;

    Particle_s_mask_r = imrotate3(uint8(Particle_s_mask),ii,[0 0 1],'linear','crop');
    Particle_se_mask_r = imrotate3(uint8(Particle_se_mask),ii,[0 0 1],'linear','crop');
    Particle_zn_mask_r = imrotate3(uint8(Particle_zn_mask),ii,[0 0 1],'linear','crop');

%x = r .* cos(elevation) .* cos(azimuth)
%y = r .* cos(elevation) .* sin(azimuth)
%z = r .* sin(elevation)

    for jj=-90:1:90
%        tmp = ((AZ >= -0 +ii) & (AZ < 1 +ii));
%        ttmp = ((EL >= -0 + jj) & (EL < 1 + jj));
        Particle_s_mask_rr = imrotate3(uint8(Particle_s_mask_r),jj,[-1 1 0],'linear','crop');
        Particle_se_mask_rr = imrotate3(uint8(Particle_se_mask_r),jj,[-1 1 0],'linear','crop');
        Particle_zn_mask_rr = imrotate3(uint8(Particle_zn_mask_r),jj,[-1 1 0],'linear','crop');

%        tttmp = tmp & ttmp;
        %figure(22); subplot(1,3,1); imagesc(tttmp(:,:,33)); 
        %subplot(1,3,2); imagesc(squeeze(tttmp(128,:,:))); 
        %subplot(1,3,3); imagesc(squeeze(tttmp(:,128,:))); 
        % clear tmp ttmp tttmp;
%        IDX = find(tttmp == 1);
        %disp(size(IDX,1));
        thickness_s(nn,mm) = sum((Particle_s_mask_rr(IDX)));
        thickness_se(nn,mm) = sum((Particle_se_mask_rr(IDX)));
        thickness_zn(nn,mm) = sum((Particle_zn_mask_rr(IDX)));
        
        nonzero_s = find(Particle_s_mask_rr(IDX)>0);
        nonzero_se = find(Particle_se_mask_rr(IDX)>0);
        nonzero_zn = find(Particle_zn_mask_rr(IDX)>0);
        
        thickness_zn_se(nn,mm) = sum(Particle_zn_mask_rr(nonzero_se));
        thickness_zn_s(nn,mm) = sum(Particle_zn_mask_rr(nonzero_s));
 
        nn = nn + 1;
    end
    mm = mm + 1;
    %figure(13); imagesc(thickness_s); 
    disp(ii);
end

%% Code for Compensation of Geometry
thickness_s_mod = NaN(181,360);
thickness_se_mod = NaN(181,360);
thickness_zn_mod = NaN(181,360);
%correct_map = NaN(181,360);
[a b] = pol2cart(deg2rad([90:-1:0]),64);
aa = [a fliplr(a(1:end-1))];
aa = aa/max(aa(:));

for ii=1:181
    if aa(ii) < aa(2)
        tmp_s = uint8(imresize(thickness_s(ii,:),0.0001));
        tmp_se = uint8(imresize(thickness_se(ii,:),0.0001));
        tmp_zn = uint8(imresize(thickness_zn(ii,:),0.0001));      
    else
        tmp_s = uint8(imresize(thickness_s(ii,:),aa(ii)));
        tmp_se = uint8(imresize(thickness_se(ii,:),aa(ii)));
        tmp_zn = uint8(imresize(thickness_zn(ii,:),aa(ii)));
    end
    thickness_s_mod(ii,uint16((360-size(tmp_s,2))/2)+1:uint16((360-size(tmp_s,2))/2)+size(tmp_s,2)) = tmp_s;
    thickness_se_mod(ii,uint16((360-size(tmp_se,2))/2)+1:uint16((360-size(tmp_se,2))/2)+size(tmp_se,2)) = tmp_se;
    thickness_zn_mod(ii,uint16((360-size(tmp_zn,2))/2)+1:uint16((360-size(tmp_zn,2))/2)+size(tmp_zn,2)) = tmp_zn;
    
    clear tmp_s tmp_se tmp_zn;
end

%figure(11); imagesc(thickness_s_mod);
%save 12_spherical_projection_particle_1_s_se_zn_zn(se)_zn(s)_inter_linear_step_1_thr_0.003.mat thickness_s thickness_s_mod thickness_se thickness_se_mod thickness_zn thickness_zn_mod
%save 12_spherical_projection_particle_2_rot_70_s_se_zn_zn(se)_zn(s)_inter_linear_step_1_thr_0.003.mat thickness_s thickness_s_mod thickness_se thickness_se_mod thickness_zn thickness_zn_mod
