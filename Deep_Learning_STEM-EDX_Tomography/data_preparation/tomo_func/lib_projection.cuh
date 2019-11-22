//*************************************************************************//
// FUNCTION
//*************************************************************************//
 
void    RunProjection(float *pdP, float *pdI, float *pdStepImg, int *pnSizeImg,
                        float *pdStepDct, int *pnSizeDct, float *pdOffsetDct, float *pdRotDct,
                        float dStepView, int nNumView, float dDSO, float dDSD);
 
void    CreateMemoryProjection(float *pdI, int *pnSizeImg, int *pnSizeDct, int nNumView);
void    DestroyMemoryProjection();
 
void    CreateWeightProjection(float *pdStepDct, int *pnSizeDct, float dDSO, float dDSD, POS_INFO posinfo);
void    DestroyWeightProjection();
 
//*************************************************************************//
// VARIABLE
//*************************************************************************//
            
texture<float, cudaTextureType3D, cudaReadModeElementType>  texpdI;
cudaArray                                                   *arrpdI     = 0;
cudaMemcpy3DParms                                           cpyParamI   = {0};
cudaExtent                                                  sizeVolumeI;
 
 
//*************************************************************************//
 
texture<float, cudaTextureType1D, cudaReadModeElementType>  texpdWgt;
float                                                       *arrpdWgt   = 0;
 
//*************************************************************************//

float                                                       *gpdP       = 0;
 
//*************************************************************************//
// FUNCTION
//*************************************************************************//
 
__global__ void    Projection(float *pdP, int sIdx)
{
    int     nIdxDctX    = blockDim.x*blockIdx.x + threadIdx.x;
    int     nIdxDctY    = blockDim.y*blockIdx.y + threadIdx.y;
 
    if(nIdxDctX >= c_dctinfo.nDctX || nIdxDctY >= c_dctinfo.nDctY)     return ;
    
    float3  *pdSrc      = &(c_posinfo[sIdx].dSrc);
    float3  *pdDct      = c_posinfo[sIdx].pdDct;
    
    float   dIdxDctY    = (nIdxDctY + 0.5f)/c_dctinfo.nDctY + pdDct[nIdxDctX].z/c_imginfo.dSizeZ;
    
    if (dIdxDctY < 0.0f || dIdxDctY >= 1.0f)  return ;
 
    register float dSample  = c_geoinfo.dSample;
 
    float3  dRayDct     = pdDct[nIdxDctX] - (*pdSrc);
    dRayDct             = dSample*normalize(dRayDct);
    
    float3  dDSF_min    = (c_imginfo.dFOV_MIN - *pdSrc)/dRayDct;
    float3  dDSF_max    = (c_imginfo.dFOV_MAX - *pdSrc)/dRayDct;
    float3  dDist_min   = fminf(dDSF_min, dDSF_max);
    float3  dDist_max   = fmaxf(dDSF_min, dDSF_max);
    
    register float dmin, dmax, cnt;
    dmin        = fmaxf(dDist_min.x, dDist_min.y);
    dmax        = fminf(dDist_max.x, dDist_max.y);
    
    if ( dmin >= dmax ) return ;
    
    cnt         = dmax - dmin;
    
    dDist_min   = ((dRayDct*dmin + *pdSrc) + c_imginfo.dFOV_MAX)/(2.0f*c_imginfo.dFOV_MAX);
    dDist_max   = ((dRayDct*dmax + *pdSrc) + c_imginfo.dFOV_MAX)/(2.0f*c_imginfo.dFOV_MAX);
    
    register float3 smStep  = (dDist_max - dDist_min)/cnt;
    
    register int    loopcnt = int(cnt + 1);
    register float3 smPos   = dDist_min;
    register float  dInterp = 0.0f;
    
    for (int i = 0; i < loopcnt; i++) {
//      Edited V Offset
//         dInterp += tex3D(texpdI, smPos.y, smPos.x, (nIdxDctY + 0.5f)/c_dctinfo.nDctY + smPos.z);
//      Edited V movement
        dInterp += tex3D(texpdI, smPos.y, smPos.x, dIdxDctY);
        smPos   = smPos + smStep;
    }
 
    pdP[c_dctinfo.nDctX*nIdxDctY + nIdxDctX]    = dSample*dInterp*tex1Dfetch(texpdWgt, nIdxDctX);
    
    return ;
}
 
//*************************************************************************//
 
void    CreateMemoryProjection(float *pdI, int *pnSizeImg, int *pnSizeDct, int nNumView)
{
    int nY      = pnSizeImg[0];
    int nX      = pnSizeImg[1];
    int nZ      = pnSizeImg[2];
    
    int nSizeDct= pnSizeDct[0]*pnSizeDct[1];
 
    texpdI.channelDesc      = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    
    texpdI.addressMode[0]   = cudaAddressModeBorder;
    texpdI.addressMode[1]   = cudaAddressModeBorder;
    texpdI.addressMode[2]   = cudaAddressModeBorder;
    
    texpdI.filterMode       = cudaFilterModeLinear;
    texpdI.normalized       = true;
        
    sizeVolumeI              = make_cudaExtent(nY, nX, nZ);
    cudaMalloc3DArray(&arrpdI, &texpdI.channelDesc, sizeVolumeI);
    
    cpyParamI.extent         = sizeVolumeI;
    cpyParamI.dstArray       = arrpdI;
    cpyParamI.srcPtr         = make_cudaPitchedPtr((void *)pdI, sizeof(float)*sizeVolumeI.width, sizeVolumeI.width, sizeVolumeI.height);
    
    cudaMemcpy3D(&cpyParamI);
    cudaBindTextureToArray(texpdI, arrpdI);
    
    //*************************************************************************//
    
    cudaMalloc((void **)&gpdP, sizeof(float)*nSizeDct*nNumView);
    cudaMemset(gpdP, 0, sizeof(float)*nSizeDct*nNumView);
//     pdP                     = (float *) malloc(sizeof(float)*nSizeDct*nNumView);
//     memset(pdP, 0, sizeof(float)*nSizeDct*nNumView);
    
    return ;
}
 
void    DestroyMemoryProjection()
{
    cudaUnbindTexture(texpdI);
    cudaFreeArray(arrpdI);      arrpdI  = 0;
    
    cudaFree(gpdP);             gpdP    = 0;
    
    return ;
}
 
/*********************************************************************/
 
void    CreateWeightProjection(float *pdStepDct, int *pnSizeDct, float dDSO, float dDSD, POS_INFO posinfo)
{
    int     nDctX       = pnSizeDct[1];
    
//     float   dDctX       = pdStepDct[1];
//     float   dSizeDctXh  = (dDctX*(nDctX - 1.0f))/2.0f;
    
    float3  *pdDct      = posinfo.pdDct;
    float   *pdWgt      = 0;
    
    pdWgt   = (float *) malloc(sizeof(float)*nDctX);
    memset(pdWgt, 0 ,sizeof(float)*nDctX);
    
    cudaMalloc((void **)&arrpdWgt, sizeof(float)*nDctX);
    cudaMemset(arrpdWgt, 0, sizeof(float)*nDctX);
    
//     #pragma omp parallel for
    for (int ix = 0; ix < nDctX; ix++) {
        pdWgt[ix]   = dDSD/sqrtf((dDSO + pdDct[ix].y)*(dDSO + pdDct[ix].y) + pdDct[ix].x*pdDct[ix].x);
    }
    
    cudaMemcpy(arrpdWgt, pdWgt, sizeof(float)*nDctX, cudaMemcpyHostToDevice);
    cudaBindTexture(NULL, texpdWgt, arrpdWgt, sizeof(float)*nDctX);
    
    /*********************************************************************/
 
    free(pdWgt);    pdWgt   = 0;
    
    return ;
}
 
void    DestroyWeightProjection()
{
    cudaUnbindTexture(texpdWgt);
    cudaFree(arrpdWgt);     arrpdWgt    = 0;
}
 
void RunProjection(float *pdP, float *pdI, float *pdStepImg, int *pnSizeImg, 
                    float *pdStepDct, int *pnSizeDct, float *pdOffsetDct, float *pdRotDct,
                    float dStepView, int nNumView, float dDSO, float dDSD)
{
    IMG_INFO    imginfo;
    DCT_INFO    dctinfo;
    GEO_INFO    geoinfo;
    POS_INFO    posinfo, oriinfo;
    
    int     nSizeDct    = pnSizeDct[0]*pnSizeDct[1];
    
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
    
    cudaMemcpyToSymbol(c_imginfo,  &imginfo,    sizeof(IMG_INFO));
    cudaMemcpyToSymbol(c_dctinfo,  &dctinfo,    sizeof(DCT_INFO));
    cudaMemcpyToSymbol(c_geoinfo,  &geoinfo,    sizeof(GEO_INFO));
    
    //*************************************************************************//    
 
    oriinfo.dSrc        = make_float3(0.0f, -geoinfo.dDSO, 0.0f);
 
    //*************************************************************************//
    float   dGAMMA  = dctinfo.dDctX*(dctinfo.nDctX/2.0f - 0.5f) + dctinfo.dOffsetX;
    
    for (int i = 0; i < dctinfo.nDctX; i++) {
        oriinfo.pdDct[i]    = make_float3(geoinfo.dDSD*sinf(dGAMMA), geoinfo.dDSD*cosf(dGAMMA) - geoinfo.dDSO, 0.0f);
        dGAMMA              -= dctinfo.dDctX;
    }
    
    CreateWeightProjection(pdStepDct, pnSizeDct, geoinfo.dDSO, geoinfo.dDSD, oriinfo);
    
    //*************************************************************************//
    
    int     nThreadX = 8;
    int     nThreadY = 8;
    
    dim3    block2d(nThreadX, nThreadY);
    dim3    grid2d((dctinfo.nDctX + nThreadX - 1)/nThreadX, (dctinfo.nDctY + nThreadY - 1)/nThreadY);
 
    //*************************************************************************//
    
    cudaStream_t    *pStream    = (cudaStream_t *)  malloc(sizeof(cudaStream_t)*nStream);
    float           **h_ppdProj = (float **)        malloc(sizeof(float *)*nStream);
    float           **d_ppdProj = (float **)        malloc(sizeof(float *)*nStream);
    
//     #pragma omp parallel for
    for (int i = 0; i < nStream; i++) {
        cudaStreamCreate(&pStream[i]);
        cudaMalloc((void **)&d_ppdProj[i],      sizeof(float)*nSizeDct);
        cudaMallocHost((void **)&h_ppdProj[i],  sizeof(float)*nSizeDct);
    }
    
    //*************************************************************************//
 
    float   dBeta       = 0.0f;
    int     nViewIter   = 0;
        
    int     sInc        = 0;
    int     sIdx        = 0;
    int     sIdx_pre    = nStream - 1;
        
    for (nViewIter = 0; nViewIter < nNumView; nViewIter++) {
 
        sIdx        = sInc%nStream;
        dBeta       = geoinfo.dStepView*nViewIter;
        
        RotPos(   &posinfo.dSrc, &oriinfo.dSrc, posinfo.pdDct, oriinfo.pdDct, dBeta, dctinfo.nDctX  );
//         RotPos(   &posinfo.dSrc, &oriinfo.dSrc,
//                     posinfo.pdDct, oriinfo.pdDct, pdOffsetDct,
//                     dBeta, dBeta + 0.0f, dctinfo.nDctX, true  );
        
        cudaMemcpyToSymbolAsync(c_posinfo, &posinfo, sizeof(POS_INFO), sizeof(POS_INFO)*sIdx, cudaMemcpyHostToDevice, pStream[sIdx]);
                
        cudaMemsetAsync(d_ppdProj[sIdx], 0, sizeof(float)*nSizeDct, pStream[sIdx]);
        
        Projection<<<grid2d, block2d, 0, pStream[sIdx]>>>(d_ppdProj[sIdx], sIdx);
        
        cudaMemcpyAsync(h_ppdProj[sIdx], d_ppdProj[sIdx], sizeof(float)*nSizeDct, cudaMemcpyDeviceToHost, pStream[sIdx]);
        
        if (nViewIter > 0) {
            cudaStreamSynchronize(pStream[sIdx_pre]);
            memcpy(&pdP[nSizeDct*(nViewIter - 1)], h_ppdProj[sIdx_pre], sizeof(float)*nSizeDct);
        }
        
        sIdx_pre    = sIdx;
        sInc++;
    }
    cudaStreamSynchronize(pStream[sIdx_pre]);
    memcpy(&pdP[nSizeDct*(nViewIter - 1)], h_ppdProj[sIdx_pre], sizeof(float)*nSizeDct);
            
    //*************************************************************************//
    
    DestroyWeightProjection();
    
//     #pragma omp parallel for
    for (sIdx = 0; sIdx < nStream; sIdx++) {
        cudaStreamDestroy(pStream[sIdx]);
        cudaFree(d_ppdProj[sIdx]);
        cudaFreeHost(h_ppdProj[sIdx]);
    }
    free(pStream);      pStream     = 0;
    free(h_ppdProj);    h_ppdProj   = 0;
    free(d_ppdProj);    d_ppdProj   = 0;
    
    //*************************************************************************//
 
    return ;
}

