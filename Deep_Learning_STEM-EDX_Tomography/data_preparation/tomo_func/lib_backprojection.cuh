//*************************************************************************//
// FUNCTION
//*************************************************************************//
            
void    RunBackProjection(float *pdI, float *pdP, float *pdStepImg, int *pnSizeImg, 
                        float *pdStepDct, int *pnSizeDct, float *pdOffsetDct, float *pdRotDct,
                        float dStepView, int nNumView, float dDSO, float dDSD);
 
void    CreateMemoryBackProjection(float *pdP, int *pnSizeImg, int *pnSizeDct, int nNumView);
void    DestroyMemoryBackProjection();
 
//*************************************************************************//
// VARIABLE
//*************************************************************************//
 
float   *gpdI   = 0;
 
texture<float, cudaTextureType2DLayered> texpdP0;
texture<float, cudaTextureType2DLayered> texpdP1;
texture<float, cudaTextureType2DLayered> texpdP2;
texture<float, cudaTextureType2DLayered> texpdP3;
 
texture<float, cudaTextureType2DLayered> *ptexpdP;
 
cudaArray                               *arrpdP[nStream];
cudaMemcpy3DParms                       cpyParamP[nStream];
cudaExtent                              sizeVolumeP;
 
//*************************************************************************//
// FUNCTION
//*************************************************************************//
 
template<int T>
__global__ void BackProjection(float *pdI, int sIdx)
{
    int     nIdxX       = blockDim.x*blockIdx.x + threadIdx.x;
    int     nIdxY       = blockDim.y*blockIdx.y + threadIdx.y;
    int     nIdxZ       = blockDim.z*blockIdx.z + threadIdx.z;
    
    int     nIdx        = c_imginfo.nY*c_imginfo.nX*nIdxZ + c_imginfo.nY*nIdxX + nIdxY;
 
    if (nIdxY >= c_imginfo.nY || nIdxX >= c_imginfo.nX || nIdxZ >= c_imginfo.nZ)    return ;
    
    FOV_INFO    *pdFOV;
    register    float3  dFOV, dVecDir;
 
    register    float   dPosDctX, L, wgt, dIdxZ;
    register    float   dInterp = 0.0f, dOffsetDctX;
    
    dIdxZ           = (nIdxZ + 0.5f)/c_imginfo.nZ;
    dOffsetDctX     = (c_dctinfo.dSizeDctX - c_dctinfo.dDctX)*0.5f + c_dctinfo.dOffsetX;
    
    wgt             = c_dctinfo.dDctX*c_geoinfo.dStepView*c_geoinfo.dDSO;
    
    for (int pIdx = 0; pIdx < nPage; pIdx++) {
        pdFOV       = &c_fovinfo[nPage*sIdx + pIdx];
        
        dFOV        = (*pdFOV).dOrig + ((*pdFOV).pdE[X]*nIdxX + (*pdFOV).pdE[Y]*nIdxY);
        dVecDir     = dFOV - c_oriinfo.dSrc;
        
        dPosDctX    = (dOffsetDctX + atan2f(dVecDir.x, dVecDir.y))/c_dctinfo.dSizeDctX;
        
        L           = 1.0f/sqrtf(dVecDir.y*dVecDir.y + dVecDir.x*dVecDir.x);
//         L           = 1.0f;
 
        if      (T == 0)
            dInterp += tex2DLayered(texpdP0, dPosDctX, dIdxZ, pIdx)*(L*L);
        else if (T == 1)
            dInterp += tex2DLayered(texpdP1, dPosDctX, dIdxZ, pIdx)*(L*L);
        else if (T == 2)
            dInterp += tex2DLayered(texpdP2, dPosDctX, dIdxZ, pIdx)*(L*L);
        else if (T == 3)
            dInterp += tex2DLayered(texpdP3, dPosDctX, dIdxZ, pIdx)*(L*L);
    }
        
    pdI[nIdx]   += wgt*dInterp;
 
    return;
}
 
//*************************************************************************//
 
void CreateMemoryBackprojection(float *pdP, int *pnSizeImg, int *pnSizeDct, int nNumView)
{
    int nY          = pnSizeImg[0];
    int nX          = pnSizeImg[1];
    int nZ          = pnSizeImg[2];
    
    int nSizeImg   = nY*nX*nZ;
    
    int nDctY       = pnSizeDct[0];
    int nDctX       = pnSizeDct[1];
 
    cudaMalloc(&gpdI, sizeof(float)*nSizeImg);
    cudaMemset(gpdI, 0, sizeof(float)*nSizeImg);
    sizeVolumeP              = make_cudaExtent(nDctX, nDctY, nPage);
    
//     #pragma omp parallel for
    for (int sIdx = 0; sIdx < nStream; sIdx++) {
        
        switch (sIdx) {
            case 0: ptexpdP     = &texpdP0; break;
            case 1: ptexpdP     = &texpdP1; break;
            case 2: ptexpdP     = &texpdP2; break;
            case 3: ptexpdP     = &texpdP3; break;
        }
                        
        (*ptexpdP).channelDesc      = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        
        (*ptexpdP).addressMode[0]   = cudaAddressModeBorder;
        (*ptexpdP).addressMode[1]   = cudaAddressModeBorder;
        
        (*ptexpdP).filterMode       = cudaFilterModeLinear;
        (*ptexpdP).normalized       = true;
        
        cudaMalloc3DArray(&arrpdP[sIdx], &(*ptexpdP).channelDesc, sizeVolumeP, cudaArrayLayered);
        
        cpyParamP[sIdx].extent         = sizeVolumeP;
        cpyParamP[sIdx].dstArray       = arrpdP[sIdx];
        
        cudaBindTextureToArray((*ptexpdP), arrpdP[sIdx]);
    }
    
    return ;
}
 
void DestroyMemoryBackprojection()
{
    cudaFree(gpdI);             gpdI    = 0;
 
    for (int sIdx = 0; sIdx < nStream; sIdx++) {
        
        switch (sIdx) {
            case 0: ptexpdP     = &texpdP0; break;
            case 1: ptexpdP     = &texpdP1; break;
            case 2: ptexpdP     = &texpdP2; break;
            case 3: ptexpdP     = &texpdP3; break;
        }
        
        cudaUnbindTexture(*ptexpdP);       
        cudaFreeArray(arrpdP[sIdx]);      arrpdP[sIdx]  = 0;
    }
    
    return ;
}
 
void RunBackProjection(float *pdI, float *pdP, float *pdStepImg, int *pnSizeImg, 
                        float *pdStepDct, int *pnSizeDct, float *pdOffsetDct, float *pdRotDct,
                        float dStepView, int nNumView, float dDSO, float dDSD)
{
    IMG_INFO    imginfo;
    DCT_INFO    dctinfo;
    GEO_INFO    geoinfo;
    POS_INFO    oriinfo;
    FOV_INFO    fovinfo[nPage], stdinfo;
    
    int     nSizeDct    = pnSizeDct[0]*pnSizeDct[1];
    int     nSizeImg    = pnSizeImg[0]*pnSizeImg[1]*pnSizeImg[2];
 
    //*************************************************************************//
    
    imginfo.dY          = pdStepImg[0];
    imginfo.dX          = pdStepImg[1];
    imginfo.dZ          = pdStepImg[2];
    
    imginfo.nY          = pnSizeImg[0];
    imginfo.nX          = pnSizeImg[1];
    imginfo.nZ          = pnSizeImg[2];
    
    imginfo.dSizeY      = imginfo.dY*imginfo.nY;
    imginfo.dSizeX      = imginfo.dX*imginfo.nX;
    imginfo.dSizeZ      = imginfo.dZ*imginfo.nZ;
    
    imginfo.dFOV_MIN    = make_float3(-imginfo.dSizeX/2.0f, -imginfo.dSizeY/2.0f, 0.0f);
    imginfo.dFOV_MAX    = make_float3(+imginfo.dSizeX/2.0f, +imginfo.dSizeY/2.0f, 0.0f);
    
    //*************************************************************************//
    
    dctinfo.dDctY       = pdStepDct[0];
    dctinfo.dDctX       = pdStepDct[1];
    
    dctinfo.nDctY       = pnSizeDct[0];
    dctinfo.nDctX       = pnSizeDct[1];
    
    dctinfo.dSizeDctY   = dctinfo.dDctY*dctinfo.nDctY;
    dctinfo.dSizeDctX   = dctinfo.dDctX*dctinfo.nDctX;
    
    dctinfo.dOffsetY    = pdOffsetDct[0];
    dctinfo.dOffsetX    = pdOffsetDct[1];
        
//     memcpy(dctinfo.dOffsetDct, pdOffsetDct, sizeof(float)*2*VIEW);
//     memcpy(dctinfo.dRotDct, pdRotDct, sizeof(float)*VIEW);
    
    //*************************************************************************//
    
    geoinfo.dStepView   = dStepView;
    geoinfo.nNumView    = nNumView;
    
    geoinfo.dDSO        = dDSO;
    geoinfo.dDSD        = dDSD;
    
    geoinfo.dSample     = fminf(imginfo.dY, imginfo.dX);
    
    //*************************************************************************//    
 
    oriinfo.dSrc        = make_float3(0.0f, -geoinfo.dDSO, 0.0f);
 
    //*************************************************************************//
    float   dGAMMA      = -dctinfo.dDctX*(dctinfo.nDctX/2.0f - 0.5f);
    
    for (int i = 0; i < dctinfo.nDctX; i++) {
        oriinfo.pdDct[i]                    = make_float3(dDSD*sinf(dGAMMA), dDSD*cosf(dGAMMA) - dDSO, 0.0f);
        dGAMMA                              += dctinfo.dDctX;
    }
        
    //*************************************************************************//
    
    cudaMemcpyToSymbol(c_imginfo,  &imginfo,    sizeof(IMG_INFO));
    cudaMemcpyToSymbol(c_dctinfo,  &dctinfo,    sizeof(DCT_INFO));
    cudaMemcpyToSymbol(c_geoinfo,  &geoinfo,    sizeof(GEO_INFO));
    cudaMemcpyToSymbol(c_oriinfo,  &oriinfo,    sizeof(POS_INFO));
    
    //*************************************************************************//
    
//     stdinfo.dOrig       = make_float3(  imginfo.dFOV_MIN.x + imginfo.dX/2.0f,
//                                         imginfo.dFOV_MIN.y + imginfo.dY/2.0f,
//                                         0.0f    );
    
    stdinfo.dOrig       = make_float3(  imginfo.dFOV_MAX.x - imginfo.dX/2.0f,
                                        imginfo.dFOV_MIN.y + imginfo.dY/2.0f,
                                        0.0f    );
 
    //*************************************************************************//
    
//     stdinfo.pdE[Y]      = make_float3(0.0f,         imginfo.dY, 0.0f);
//     stdinfo.pdE[X]      = make_float3(imginfo.dX,   0.0f,       0.0f);
//     stdinfo.pdE[Z]      = make_float3(0.0f,         0.0f,       0.0f);
 
    stdinfo.pdE[Y]      = make_float3(0.0f,         imginfo.dY, 0.0f);
    stdinfo.pdE[X]      = make_float3(-imginfo.dX,   0.0f,       0.0f);
    stdinfo.pdE[Z]      = make_float3(0.0f,         0.0f,       0.0f);
    
    //*************************************************************************//
    
    int     nThreadY    = 8;
    int     nThreadX    = 4;
    int     nThreadZ    = 4;
    
    dim3    block3d(nThreadX, nThreadY, nThreadZ);
    dim3    grid3d( (imginfo.nX + nThreadX - 1)/nThreadX, 
                    (imginfo.nY + nThreadY - 1)/nThreadY,
                    (imginfo.nZ + nThreadZ - 1)/nThreadZ    );
        
    //*************************************************************************//
    
    cudaStream_t    *pStream    = (cudaStream_t *)  malloc(sizeof(cudaStream_t)*nStream);
    
//     #pragma omp parallel for
    for (int i = 0; i < nStream; i++) {
        cudaStreamCreate(&pStream[i]);
    }
    
    //*************************************************************************//
    
    float   dBeta       = 0.0f;
    int     nViewIter   = 0;
        
    int     sInc        = 0;
    int     sIdx        = 0;
    int     sIdx_pre    = nStream - 1;
    
    for(nViewIter = 0; nViewIter < nNumView; nViewIter += nPage) {
        sIdx        = sInc%nStream;
        
        cpyParamP[sIdx].srcPtr         = make_cudaPitchedPtr((void *)&pdP[nSizeDct*nViewIter], sizeof(float)*sizeVolumeP.width, sizeVolumeP.width, sizeVolumeP.height);
        cudaMemcpy3DAsync(&cpyParamP[sIdx], pStream[sIdx]);
        
//         #pragma omp parallel for
        for (int nPageIter = 0; nPageIter < nPage; nPageIter++) {
            dBeta       = -geoinfo.dStepView*(nViewIter + nPageIter);
            RotFOV(   &fovinfo[nPageIter].dOrig, &stdinfo.dOrig, fovinfo[nPageIter].pdE, stdinfo.pdE, -dBeta  );
        }
        cudaMemcpyToSymbolAsync(c_fovinfo, fovinfo, sizeof(FOV_INFO)*nPage, sizeof(FOV_INFO)*(nPage*sIdx), cudaMemcpyHostToDevice, pStream[sIdx]);
 
        switch (sIdx) {
            case 0: BackProjection<0><<<grid3d, block3d, 0, pStream[sIdx]>>>(gpdI, sIdx);   break;
            case 1: BackProjection<1><<<grid3d, block3d, 0, pStream[sIdx]>>>(gpdI, sIdx);   break;
            case 2: BackProjection<2><<<grid3d, block3d, 0, pStream[sIdx]>>>(gpdI, sIdx);   break;
            case 3: BackProjection<3><<<grid3d, block3d, 0, pStream[sIdx]>>>(gpdI, sIdx);   break;
        }
 
        sIdx_pre    = sIdx;
        sInc++;
    }
    
    //*************************************************************************//
    
//     #pragma omp parallel for
    for (sIdx = 0; sIdx < nStream; sIdx++)  cudaStreamSynchronize(pStream[sIdx]);
    cudaMemcpy(pdI, gpdI, sizeof(float)*nSizeImg, cudaMemcpyDeviceToHost);
    
//     #pragma omp parallel for
    for (sIdx = 0; sIdx < nStream; sIdx++)  cudaStreamDestroy(pStream[sIdx]);
    free(pStream);      pStream     = 0;
    
    return ;
}

