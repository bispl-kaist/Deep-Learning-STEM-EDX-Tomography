function param = MakeParam(pdSizeImg, pnSizeImg, dStepDct, pnSizeDct, dOffset, dStepView, nNumView, DSO, DSD)

param.DSO       = DSO;
param.DSD       = DSD;

param.dY        = pdSizeImg(1)/pnSizeImg(1);
param.dX        = pdSizeImg(2)/pnSizeImg(2);
param.dZ        = pdSizeImg(3)/pnSizeImg(3);

param.nY        = pnSizeImg(1);
param.nX        = pnSizeImg(2);
param.nZ        = pnSizeImg(3);

param.dDctX     = dStepDct;
param.nDctY     = pnSizeDct(1);
param.nDctX     = pnSizeDct(2);
param.dOffsetX  = dStepDct*dOffset;

param.dStepView = dStepView;
param.nNumView  = nNumView;

end