#include "mex.h"
#include "cuda.h"

#include <string.h>
#include <math.h>

//*************************************************************************//
// FUNCTION
//*************************************************************************//

__global__ void     Projection(float *pdP, float *pdI, float dBeta, float dY, float dX, float dZ, int nY, int nX, int nZ,
                            float dStepDct, int nDctY, int nDctX, float dOffset, float dStepView, float dDSO, float dRadius, float dSample, int nNumSample);

__device__ void     Rot(float &dRotPosX, float &dRotPosY, float dPosX, float dPosY, float dTheta);
__device__ void     Rot(float &dPosX, float &dPosY, float dTheta);

//*************************************************************************//

            void    RunProjection(float *pdP, float *pdI, float *pdStepImg, int *pnSizeImg,
                                float dStepDct, int *pnSizeDct, float dOffset, float dStepView, int nNumView, float dDSO, float dDSD);

            void    CreateMemory(float *pdI, int *pnSizeImg, int *pnSizeDct, int nNumView);
            void    DestroyMemory();
            
//*************************************************************************//
// VARIABLE
//*************************************************************************//
            
//     float   *gpdP	= 0;
// 
//     texture<float, cudaTextureType2D, cudaReadModeElementType> texpdI;
//     cudaArray   *arrpdI   = 0;    
    
    
     
//*************************************************************************//
// VARIABLE
//*************************************************************************//
            
texture<float, cudaTextureType3D, cudaReadModeElementType>  texpdI;
cudaArray                                                   *arrpdI     = 0;
cudaMemcpy3DParms                                           cpyParamI   = {0};
cudaExtent                                                  sizeVolumeI;
 
 
//*************************************************************************//
 
// texture<float, cudaTextureType1D, cudaReadModeElementType>  texpdWgt;
// float                                                       *arrpdWgt   = 0;
 
//*************************************************************************//

float                                                       *gpdP       = 0;


    
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
    
    //*************************************************************************//
    
    mwSize  ndim            = 3;
    mwSize  dims[3]         = {pnSizeDct[1], pnSizeDct[0], nNumView};
    
    plhs[0]                 = mxCreateNumericArray(ndim, (const mwSize *)dims, mxSINGLE_CLASS, mxREAL);
    
    float *pdDst            = (float *) mxGetData(plhs[0]);
    memset(pdDst, 0, sizeof(float)*pnSizeDct[0]*pnSizeDct[1]*nNumView);
    
    //*************************************************************************//
    
    CreateMemory(pdSrc, pnSizeImg, pnSizeDct, nNumView);

    //*************************************************************************//

    RunProjection(pdDst, pdSrc, pdStepImg, pnSizeImg, dStepDct, pnSizeDct, dOffset, dStepView, nNumView, dDSO, dDSD);
    
    //*************************************************************************//

    DestroyMemory();
    
    //*************************************************************************//
    
    return ;
}

void RunProjection(float *pdP, float *pdI, float *pdStepImg, int *pnSizeImg, float dStepDct, int *pnSizeDct, float dOffset, float dStepView, int nNumView, float dDSO, float dDSD)
{
    float   dY      = pdStepImg[0];
    float   dX      = pdStepImg[1];
    float   dZ      = pdStepImg[2];
    
    int     nY      = pnSizeImg[0];
    int     nX      = pnSizeImg[1];
    int     nZ      = pnSizeImg[2];
    
    int     nDctY   = pnSizeDct[0];
    int     nDctX   = pnSizeDct[1];
    
    float   dSizeY  = dY*nY;
    float   dSizeX  = dX*nX;
    
    int     nSizeDct= nDctY*nDctX;
    
    //*************************************************************************//
    
//     dStepDct        = dStepDct*dDSO/dDSD;
    
    //*************************************************************************//
    
    int     nThread = 8;
    
    dim3    block;
    dim3    grid;
    
    block.x     = nThread;
    grid.x      = (int)ceil(float(nDctX)/nThread);
    
    block.y     = nThread;
    grid.y      = (int)ceil(float(nDctY)/nThread);

    float   dBeta       = 0.0f;
    
    float   dRadius     = 0.5f*sqrtf(dSizeY*dSizeY + dSizeX*dSizeX);
    float   dDiameter   = 2.0f*dRadius;
    
    float   dSample     = min(dY, dX);
    int     nNumSample  = ceil(dDiameter/dSample);
    
    //*************************************************************************//
    
    for (int nViewIter = 0; nViewIter < nNumView; nViewIter++) {
        
        dBeta   = dStepView*nViewIter;
        
        Projection<<<grid, block>>>(&gpdP[nSizeDct*nViewIter], 0, dBeta, dY, dX, dZ, nY, nX, nZ, dStepDct, nDctY, nDctX, dOffset, dStepView, dDSO, dRadius, dSample, nNumSample);
        
        cudaThreadSynchronize();
    }
    
    //*************************************************************************//
    
    cudaMemcpy(pdP, gpdP, sizeof(float)*nSizeDct*nNumView, cudaMemcpyDeviceToHost);
    
    return ;
}

__global__ void    Projection(float *pdP, float *pdI, float dBeta, float dY, float dX, float dZ, int nY, int nX, int nZ,
                            float dStepDct, int nDctY, int nDctX, float dOffset, float dStepView, float dDSO, float dRadius, float dSample, int nNumSample)
{
    int     nIdxDctX	= blockDim.x*blockIdx.x + threadIdx.x;
    int     nIdxDctY    = blockDim.y*blockIdx.y + threadIdx.y;
    int     nIdxDct    	= nDctX*nIdxDctY + nIdxDctX;


    if(nIdxDctX >= nDctX)	return ;
    if(nIdxDctY >= nDctY)      return ;


    //*************************************************************************//
    
    float	dPosDct     = dStepDct*(-(nDctX - 1.0f)/2.0f + nIdxDctX) + dOffset;
//     float   dGamma      = atan2f(dPosDct, dDSO);
    float   dGamma      = 0;
    
    //*************************************************************************//

    float   dSampleY    = dSample*__cosf(dGamma);
    float   dSampleX    = dSample*__sinf(dGamma);
    
    float   dPosY       = -dRadius;
//     float   dPosX       = (dDSO - dRadius)*__tanf(dGamma);
    float   dPosX       = dPosDct;
   
    Rot(dSampleX, dSampleY, dBeta);
    Rot(dPosX, dPosY, dBeta);

    float   dMaxY       = dY*nY/2.0f;
    float   dMaxX       = dX*nX/2.0f;
    float   dMaxZ       = dZ*nZ/2.0f;
        
    //*************************************************************************//
    
//     float COS_THETA	= __cosf(dBeta);
//     float SIN_THETA	= __sinf(dBeta);
        
    //*************************************************************************//
    
    float   dInterp     = 0.0f;
    
    for (int nIdxSample = 0; nIdxSample < nNumSample; nIdxSample++) {

//         Rot(dRotPosX, dRotPosY, dPosX, dPosY, dBeta);
        
//         dRotPosX        = COS_THETA*dPosX + SIN_THETA*dPosY;    // X Pos
//         dRotPosY        = -SIN_THETA*dPosX + COS_THETA*dPosY;   // Y Pos

        dInterp         += tex3D(texpdI, (dPosY + dMaxY)/dY, (dPosX + dMaxX)/dX, (nIdxDctY + 0.5));
        
        dPosY           += dSampleY;
        dPosX           += dSampleX;
    }
    
    pdP[nIdxDct]        = dSample*dInterp;
//     pdP[nIdxDct]        = dPosY/dMaxY + 0.5f;
    
    return ;
}

__device__ void Rot(float &dRotPosX, float &dRotPosY, float dPosX, float dPosY, float dTheta)
{
    float COS_THETA	= __cosf(dTheta);
    float SIN_THETA	= __sinf(dTheta);
    
    dRotPosX	= COS_THETA*dPosX - SIN_THETA*dPosY;    // X Pos
    dRotPosY    = SIN_THETA*dPosX + COS_THETA*dPosY;   // Y Pos
    
    return ;
}

__device__ void Rot(float &dPosX, float &dPosY, float dTheta)
{
    float COS_THETA	= __cosf(dTheta);
    float SIN_THETA	= __sinf(dTheta);
    
    float dOriX     = dPosX;
    float dOriY     = dPosY;
    
    dPosX           = COS_THETA*dOriX - SIN_THETA*dOriY;    // X Pos
    dPosY           = SIN_THETA*dOriX + COS_THETA*dOriY;   // Y Pos
    
    return ;
}

void    CreateMemory(float *pdI, int *pnSizeImg, int *pnSizeDct, int nNumView)
{
    int nY      = pnSizeImg[0];
    int nX      = pnSizeImg[1];
    int nZ      = pnSizeImg[2];
    
    int nDctY	= pnSizeDct[0];
    int nDctX	= pnSizeDct[1];
    
    int nSizeDct = nDctY*nDctX;
    
    texpdI.channelDesc      = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    
    texpdI.addressMode[0]   = cudaAddressModeBorder;
    texpdI.addressMode[1]   = cudaAddressModeBorder;
    texpdI.addressMode[2]   = cudaAddressModeBorder;
    
    texpdI.filterMode       = cudaFilterModeLinear;
    texpdI.normalized       = false;
    
    sizeVolumeI             = make_cudaExtent(nY, nX, nZ);
    cudaMalloc3DArray(&arrpdI, &texpdI.channelDesc, sizeVolumeI);
    
    cpyParamI.extent        = sizeVolumeI;
    cpyParamI.dstArray      = arrpdI;
    cpyParamI.srcPtr        = make_cudaPitchedPtr((void *)pdI, sizeof(float)*sizeVolumeI.width, sizeVolumeI.width, sizeVolumeI.height);
    
    cudaMemcpy3D(&cpyParamI);
    cudaBindTextureToArray(texpdI, arrpdI);
    
    //*************************************************************************//
    cudaMalloc((void **)&gpdP, sizeof(float)*nSizeDct*nNumView);
    cudaMemset(gpdP, 0, sizeof(float)*nSizeDct*nNumView);

    
    return ;
}

void    DestroyMemory()
{
    cudaUnbindTexture(texpdI);
    cudaFreeArray(arrpdI);      arrpdI      = 0;
    
    cudaFree(gpdP);             gpdP        = 0;
    
    return ;
}