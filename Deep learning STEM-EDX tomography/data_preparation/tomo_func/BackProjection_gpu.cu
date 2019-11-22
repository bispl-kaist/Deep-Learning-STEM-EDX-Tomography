#include "mex.h"
#include "cuda.h"

#include <string.h>
#include <math.h>

//*************************************************************************//
// FUNCTION
//*************************************************************************//

__global__ void     BackProjection(float *pdI, float *pdP, float dBeta, float dY, float dX, float dZ, int nY, int nX, int nZ,
                                float dStepDct, int nDctY, int nDctX, float dOffset, float dStepView, float dDSO);

__device__  int     Pos2Idx(float dPos, float dStep, int nSize);
__device__  float   Idx2Pos(int nIdx, float dStep, int nSize);

__device__  float	Interp1D(float *pdSrc, float dPos, int nSize, float dStep);

//*************************************************************************//
            
            void    RunBackProjection(float *pdI, float *pdP, float *pdStepImg, int *pnSizeImg, 
                                    float dStepDct, int *pnSizeDct, float dOffset, float dStepView, int nNumView, float dDSO, float dDSD);
            
            void    CreateMemory(float *pdP, int *pnSizeImg, int *pnSizeDct, int nNumView);
            void    DestroyMemory();
            
//*************************************************************************//
// VARIABLE
//*************************************************************************//

#define	NPAGE   1

float   *gpdP	= 0;
float	*gpdI	= 0;

texture<float, cudaTextureType2DLayered>	texpdP;
cudaArray                                   *arrpdP     = 0;
cudaMemcpy3DParms                           cpyParamP   = {0};
cudaExtent                                  sizeVolumeP;
            
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float	*pdSrc          = (float *) mxGetData(prhs[0]);
    
    float   *pdStepImg      = (float *) mxGetData(prhs[1]);
    int     *pnSizeImg      = (int *)   mxGetData(prhs[2]);
    
    float   dStepDct        = (float)   mxGetScalar(prhs[3]);
    int     *pnSizeDct      = (int *)   mxGetData(prhs[4]);
    
    float   dOffset         = (float)   mxGetScalar(prhs[5]);
    
    float   dStepView        = (float)  mxGetScalar(prhs[6]);
    int     nNumView         = (int)    mxGetScalar(prhs[7]);
    
    float   dDSO            = (float)   mxGetScalar(prhs[8]);
    float   dDSD            = (float)   mxGetScalar(prhs[9]);

    mwSize  ndim            = 3;
    mwSize  dims[3]         = {pnSizeImg[0], pnSizeImg[1], pnSizeImg[2]};
    
    plhs[0]                 = mxCreateNumericArray(ndim, (const mwSize *)dims, mxSINGLE_CLASS, mxREAL);
    float *pdDst            = (float *) mxGetData(plhs[0]);
    
    //*************************************************************************//
    
    CreateMemory(pdSrc, pnSizeImg, pnSizeDct, nNumView);
    
    //*************************************************************************//
    
    RunBackProjection(pdDst, pdSrc, pdStepImg, pnSizeImg, dStepDct, pnSizeDct, dOffset, dStepView, nNumView, dDSO, dDSD);
    
    //*************************************************************************//
    
    DestroyMemory();
    
    return ;
}

void RunBackProjection(float *pdI, float *pdP, float *pdStepImg, int *pnSizeImg, float dStepDct, int *pnSizeDct, float dOffset, float dStepView, int nNumView, float dDSO, float dDSD)
{
    float   dBeta       = 0.0f;
    
    float   dY          = pdStepImg[0];
    float   dX          = pdStepImg[1];
    float   dZ          = pdStepImg[2];
    
    int     nY          = pnSizeImg[0];
    int     nX          = pnSizeImg[1];
    int     nZ          = pnSizeImg[2];
    
    int     nDctY       = pnSizeDct[0];
    int     nDctX       = pnSizeDct[1];
    
    int     nSizeImg	= nY*nX*nZ;
    int     nSizeDct    = nDctY*nDctX;
    
    //*************************************************************************//
    
//     dStepDct            = dStepDct*dDSO/dDSD;
    
    //*************************************************************************//
    
    int     nThread	= 8;
    
    dim3    block;
    dim3    grid;
    
    block.x	= nThread;
    block.y = nThread;
    block.z = nThread;
    
    grid.x  = (int)ceil(float(nX)/nThread);
    grid.y  = (int)ceil(float(nY)/nThread);
    grid.z  = (int)ceil(float(nZ)/nThread);
    
    //*************************************************************************//
    
    for(int nViewIter = 0; nViewIter < nNumView; nViewIter += NPAGE) {
        
        dBeta   = dStepView*nViewIter;
        
        cpyParamP.srcPtr         = make_cudaPitchedPtr((void *)&pdP[nSizeDct*nViewIter], sizeof(float)*sizeVolumeP.width, sizeVolumeP.width, sizeVolumeP.height);
        cudaMemcpy3D(&cpyParamP);

        BackProjection<<<grid, block>>>(gpdI, 0, dBeta, dY, dX, dZ, nY, nX, nZ, dStepDct, nDctY, nDctX, dOffset, dStepView, dDSO);
        cudaThreadSynchronize();

    }
    
    //*************************************************************************//
    
    cudaMemcpy(pdI, gpdI, sizeof(float)*nSizeImg, cudaMemcpyDeviceToHost);
    
    return ;
}

__global__ void BackProjection(float *pdI, float *pdP, float dBeta, float dY, float dX, float dZ, int nY, int nX, int nZ,
                                float dStepDct, int nDctY, int nDctX, float dOffset, float dStepView, float dDSO)
{
    int     nIdxX       = blockDim.x*blockIdx.x + threadIdx.x;
    int     nIdxY       = blockDim.y*blockIdx.y + threadIdx.y;
    int     nIdxZ       = blockDim.z*blockIdx.z + threadIdx.z;

    if (nIdxX >= nX)	return ;
    if (nIdxY >= nY)	return ;
    if (nIdxZ >= nZ)	return ;
    
    int     nIdx        = nY*nX*nIdxZ + nY*nIdxX + nIdxY;
    
    //*************************************************************************//

    float   dMaxU       = dStepDct*nDctX/2.0f;
    
    float   dPosX       = dX*(-(nX - 1.0f)/2.0f + nIdxX);
    float   dPosY       = dY*(-(nY - 1.0f)/2.0f + nIdxY);
    
    float   dRadius     = sqrtf(dPosX*dPosX + dPosY*dPosY);
    float   dPhi        = atan2f(dPosY, dPosX);
    
//     float   dDistX      = dDSO - dRadius*__sinf(dBeta - dPhi);
//     float   dDistY      = dRadius*__cosf(dBeta - dPhi);
    float   dDistY      = 0.0f;
    
//     float   dGamma      = atan2f(dDistY, dDistX);
    float   dGamma      = 0.0f;
//     float   dDistU      = dDSO*__tanf(dGamma) + dOffset;
//     float   dDistU      = dDistY + dOffset;
    float   dDistU      = 0.0f;
    
    if(fabsf(dDistU) >= dMaxU)	return;

//     float   dInterp     = Interp1D(pdP, dDistU, nNumDct, dStepDct)*dStepDct*dStepView;
//     float   U           = dDistX/dDSO;
    
    float   dInterp     = 0.0f;
    
    for (int pIdx = 0; pIdx < NPAGE; pIdx++) {
        dDistY	= dRadius*__cosf(dBeta + dStepView*pIdx - dPhi);
        dDistU  = (dDistY + dOffset + dMaxU)/dStepDct;
        dInterp += tex2DLayered(texpdP, dDistU, (nIdxZ + 0.5), pIdx);
    }
    
//     pdI[nIdx]           += 0.5f*__cosf(dGamma)*dInterp/(U*U);
//     pdI[nIdx]           += 0.5f*dInterp*dStepDct*dStepView;
    pdI[nIdx]           += 0.25f*dInterp*dStepDct*dStepView;
    
//     pdI[nIdx]   = dDistU;

    return;
}

__device__ float Interp1D(float *pdSrc, float dPos, int nSize, float dStep)
{
    float	dDst        = 0;
    
    float	pdWgt[2]    = {0, 0};
    
    int     pnIdx[2]    = {0, 0};
    float	pdPos[2]    = {0, 0};
    
    float	pdValue[2]	= {0, 0};
    
    pnIdx[0]= Pos2Idx(dPos, dStep, nSize);
    pnIdx[1]= pnIdx[0] + 1;
    
    pdPos[0]= Idx2Pos(pnIdx[0], dStep, nSize);
    pdPos[1]= pdPos[0] + dStep;
    
    pdWgt[0]= (pdPos[1] - dPos)/dStep;
    pdWgt[1]= 1.0 - pdWgt[0];
    
    if(pnIdx[0] >= 0 && pnIdx[0] < nSize)       pdValue[0]  = pdSrc[pnIdx[0]];
    else                                        pdValue[0]  = 0;
    
    if(pnIdx[1] >= 0 && pnIdx[1] < nSize)       pdValue[1]  = pdSrc[pnIdx[1]];
    else                                        pdValue[1]  = 0;
    
    dDst	= pdWgt[0]*pdValue[0] + pdWgt[1]*pdValue[1];
    
    return dDst;
}

__device__ int  Pos2Idx(float dPos, float dStep, int nSize)
{
    int nIdx    = 0;
    nIdx        = int(dPos/dStep + (nSize - 1.0f)/2.0f);
    
    return nIdx;
}

__device__ float Idx2Pos(int nIdx, float dStep, int nSize)
{
    float dPos     = 0;
    dPos            = (nIdx - (nSize - 1.0f)/2.0f)*dStep;
    
    return dPos;
}

void CreateMemory(float *pdP, int *pnSizeImg, int *pnSizeDct, int nNumView)
{
    int nY          = pnSizeImg[0];
    int nX          = pnSizeImg[1];
    int nZ          = pnSizeImg[2];
    
    int nDctY       = pnSizeDct[0];
    int nDctX       = pnSizeDct[1];
    
    int nSizeImg	= nY*nX*nZ;
    
    cudaMalloc(&gpdI, sizeof(float)*nSizeImg);
    cudaMemset(gpdI, 0, sizeof(float)*nSizeImg);
    sizeVolumeP             = make_cudaExtent(nDctX, nDctY, NPAGE);
    
    texpdP.channelDesc      = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    texpdP.addressMode[0]   = cudaAddressModeBorder;
    texpdP.addressMode[1]   = cudaAddressModeBorder;
    
    texpdP.filterMode       = cudaFilterModeLinear;
    texpdP.normalized       = false;
    
    cudaMalloc3DArray(&arrpdP, &texpdP.channelDesc, sizeVolumeP, cudaArrayLayered);
    
    cpyParamP.extent        = sizeVolumeP;
    cpyParamP.dstArray      = arrpdP;
    
    cudaBindTextureToArray(texpdP, arrpdP);
    
    
    
//     %%
//     cudaMalloc(&gpdP, sizeof(float)*nNumDct*nNumView);   cudaMemset(gpdP, 0, sizeof(float)*nNumDct*nNumView);
//         
//     cudaMemcpy(gpdP, pdP, sizeof(float)*nNumDct*nNumView, cudaMemcpyHostToDevice);
    
    return ;
}

void DestroyMemory()
{
    cudaFree(gpdI);         gpdI    = 0;
//     cudaFree(gpdP);     gpdP    = 0;
    
    cudaUnbindTexture(texpdP);
    cudaFreeArray(arrpdP);  arrpdP  = 0;
    
    return ;
}