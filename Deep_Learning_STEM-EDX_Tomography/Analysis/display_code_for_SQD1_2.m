load('colormap.mat');

% SQD1-2
load('8_spherical_projection_particle_2_s_se_zn_zn(se)_zn(s)_inter_linear_step_1_thr_0.003.mat')
thickness_s_1 = thickness_s_mod;
thickness_se_1 = thickness_se_mod;
thickness_zn_1 = thickness_zn_mod;
figure; 
thickness_s_bw = zeros(181,360);
IDX = find(thickness_s_1(:)==0);
thickness_s_bw(IDX) = 1;
IDX2 = find(thickness_s_1(:)>0);
thickness_s_bw(IDX2) = 0;
[tmp_bd tmp_bd_l] = bwboundaries(thickness_s_bw);
B = tmp_bd;

ax1=subplot(3,2,1); imagesc([0:359],[-90:90],(thickness_se_1)); axis image; colormap(ax1,red_map); %title([mean(thickness_se(:)) std(thickness_se(:))]); 
xlabel('Phi(\Phi)'); ylabel('Theta(\theta)'); title('SQD1-2 Se map'); colorbar;
ax2=subplot(3,2,3); imagesc([0:359],[-90:90],(thickness_s_1)); axis image; colormap(ax2, green_map); %title([mean(thickness_s(:)) std(thickness_s(:))]);
xlabel('Phi(\Phi)'); ylabel('Theta(\theta)'); title('SQD1-2 S map'); colorbar;
hold on
for k = 1:length(B)
boundary = B{k};
plot(boundary(:,2), boundary(:,1)-90, 'r', 'LineWidth', 2)
end
ax3=subplot(3,2,5); imagesc([0:359],[-90:90],(thickness_zn_1)); axis image; colormap(ax3,blue_map); %title([mean(thickness_zn(:)) std(thickness_zn(:))]);
xlabel('Phi(\Phi)'); ylabel('Theta(\theta)'); title('SQD1-2 Zn map'); colorbar;

subplot(3,2,2); histogram(thickness_se_1(:),[0:45]); title('sQD1-2(Se)'); ylabel('Count'); xlabel('Thickness (Pixel)');
subplot(3,2,4); histogram(thickness_s_1(:),[0:45]); title('sQD1-2(S)'); ylabel('Count'); xlabel('Thickness (Pixel)');
subplot(3,2,6); histogram(thickness_zn_1(:),[0:45]); title('sQD1-2(Zn)'); ylabel('Count'); xlabel('Thickness (Pixel)');

IDX1 = find(~isnan(thickness_se_mod(:)));
mean_se = mean(thickness_se_1(IDX1));
std_se = std(thickness_se_1(IDX1));
IDX2 = find(~isnan(thickness_s_1(:)));
mean_s = mean(thickness_s_1(IDX2));
std_s = std(thickness_s_1(IDX2));
IDX3 = find(thickness_s_1==0);
nonzero_s = 100*size(IDX3,1)/size(find(~isnan(thickness_s_1(:))),1);
IDX4 = find(~isnan(thickness_zn_1(:)));
mean_zn = mean(thickness_zn_1(IDX4));
std_zn = std(thickness_zn_1(IDX4));

formatSpec = "The Statistical Values (Se, S, Zn, Non Sulfur Area) of SQD1-2 are  %2.1f\x00B1%1.1f, %2.1f\x00B1%1.1f, %2.1f\x00B1%1.1f, %2.1f.";
sprintf(formatSpec, mean_se, std_se, mean_s,std_s,mean_zn,std_zn, nonzero_s)



