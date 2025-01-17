function FltP   = Filtering(P, param)

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


FltP        = single(Filtering_gpu(single(P), single(pdStepImg), int32(pnSizeImg), single(dStepDct), int32(pnSizeDct), single(dOffset), single(dStepView), int32(nNumView), single(DSO), single(DSD)));

end