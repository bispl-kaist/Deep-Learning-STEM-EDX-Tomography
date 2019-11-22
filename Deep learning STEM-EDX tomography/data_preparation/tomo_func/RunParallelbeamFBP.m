clear ;
close all;

reset(gpuDevice(1));

%% COMPILE
% MexCompile();

%% Parameter Setting
DSO             = 400;              % [mm]
DSD             = 1000;             % [mm]

%% Make Object
img             = repmat(single(phantom(512)), [1, 1, 12]);

pdImgSize       = [512, 512, 12];   % [mm x mm]
pnImgSize       = size(img);        % [elements x elements]


%% Make Detector
dStepDct        = 1;              % [mm]
pnSizeDct       = [pnImgSize(3), 1440];       % [elements]

dOffset         = 0;                % [elements]

%% Rotation Setup
nNumView        = 1440;             % [elements]
dStepView       = 2*pi/nNumView;	% [radian]
% dStepView       = pi/nNumView;      % [radian]

%% Make Object (Image, Detector)
tic;
param           = MakeParam(pdImgSize, pnImgSize, dStepDct, pnSizeDct, dOffset, dStepView, nNumView, DSO, DSD);
toc;

%% Projection
tic;
P               = Projection(img, param);
toc;

%% Visualize projection
% for iview = 1:nNumView
%     figure(1); imagesc(P(:,:,iview));
%     pause();
% end

%% Filtering
tic;
FltP            = Filtering(P, param);
toc;

% for iview = 1:nNumView
%     figure(1); imagesc(FltP(:,:,iview));    colorbar;
%     drawnow();
% end
% 
% return ;


%% BackProjection
tic;
I               = BackProjection(FltP, param);
toc;

%% Visualize backprojection
for i = 1:12
    figure(1); imagesc(I(:,:,i))
    pause();
end

%%

iz              = 12;
wndVal          = [min(img(:)), max(img(:))];

figure; 
subplot(2,3,1);     imagesc(squeeze(P(:, iz, :)));             colormap gray;              title('PROJECTION');
subplot(2,3,4);     imagesc(squeeze(FltP(:, iz, :)));          colormap gray;              title('FILTERED PROJECTION');
subplot(2,3,2);     imagesc(img(:,:,iz), wndVal);   colormap gray; axis image;  title('GROUND TRUTH');
subplot(2,3,3);     imagesc(I(:,:,iz), wndVal);     colormap gray; axis image;  title('RECON');
subplot(2,3,5);     imagesc(I(:,:,iz) - img(:,:,iz));       colormap gray; axis image;  title('GR - REC');
subplot(2,3,6);     plot(img(:,end/2,iz));  hold on;
                    plot(I(:,end/2,iz));    hold off;
xlim([1, pnImgSize(2)]);
immse(img(:,:,iz), I(:,:,iz))

return ;

