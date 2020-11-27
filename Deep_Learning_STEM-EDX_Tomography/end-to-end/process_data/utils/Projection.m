function P  = Projection(img, param)

%% Initialize

DSO         = param.DSO;
DSD         = param.DSD;

dStepDct	= param.dDctX;
dOffset     = param.dOffsetX;

dStepView	= param.dStepView;
nNumView    = param.nNumView;

pnSizeImg	= [param.nY, param.nX, param.nZ];
pdStepImg	= [param.dY, param.dX, param.dZ];

pnSizeDct   = [param.nDctY, param.nDctX];

% tic;

P	= single(Projection_gpu(single(img), single(pdStepImg), int32(pnSizeImg), single(dStepDct), int32(pnSizeDct), single(dOffset), single(dStepView), int32(nNumView), single(DSO), single(DSD)));
% P	= single(Projection_gpu_hj(single(img), single(pdStepImg), int32(pnSizeImg), single(dStepDct), int32(pnSizeDct), single(dOffset), single(dStepView), int32(nNumView), single(DSO), single(DSD)));

% toc;

end
