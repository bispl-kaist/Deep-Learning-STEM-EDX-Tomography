#include    <stdio.h>
#include    <iostream>
#include    <string.h>
#include    <cmath>

#include    "mex.h"
#include    "cuda.h"

#define     PI          3.14159265359f

//*************************************************************************//
// FUNCTION
//*************************************************************************//

__global__  void conv(float *pdDst, float *pdSrc, float *pdKernel, int nMY, int nMX, int nMZ, int nN);

//*************************************************************************//

            float sinc(float dPos);
            void Filter(float *pdFlt, float dS, int nSize, float dDSO, float dDSD);
            void Filtering(float *pdDst, float *pdSrc, float *pdKernel, int *pnSizenDct, int nNumView);
            
//*************************************************************************//
            
            void CreateMemory(float *pdSrc, int *pnSizeDct, int nNumView);
            void DestroyMemory();

//*************************************************************************//
// VARIABLE
//*************************************************************************//

float *gpdP     = 0;
float *gpdFltP  = 0;

float *gpdFlt   = 0;
float *pdFlt    = 0;
            
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
    
    plhs[0]             = mxCreateNumericArray(ndim, (const mwSize *)dims, mxSINGLE_CLASS, mxREAL);
    
    float *pdDst        = (float *)  mxGetPr(plhs[0]);
    memset(pdDst, 0, sizeof(float)*pnSizeDct[0]*pnSizeDct[1]*nNumView);

    //*************************************************************************//

    CreateMemory(pdSrc, pnSizeDct, nNumView);
    
    //*************************************************************************//
    
    Filter(pdFlt, dStepDct, pnSizeDct[1], dDSO, dDSD);
    Filtering(pdDst, pdSrc, pdFlt, pnSizeDct, nNumView);
    
    //*************************************************************************//
    
    DestroyMemory();
    
    //*************************************************************************//
    
    return ;
}

float sinc(float dPos)
{
    float   dVal    = 0.0f;
    
    if (dPos)	dVal = sinf(PI*dPos)/(PI*dPos);
    else        dVal = 1.0f;
    
    return dVal;
}

void Filter(float *pdFlt, float dS, int nSize, float dDSO, float dDSD)
{
//     dS          = dS*dDSO/dDSD;
    
    int nCnt    = 0;
    
    for (int i = -(nSize - 1); i <= (nSize - 1); i++) {
//         pdFlt[nCnt] = 1.0f/(2.0f*dS*dS)*sinc(float(i - 0.5f)) - 1.0f/(4.0f*dS*dS)*sinc(float(i - 0.5f)/2.0f)*sinc(float(i - 0.5f)/2.0f);
        pdFlt[nCnt] = 2.0f*(1.0f/(2.0f*dS*dS)*sinc(float(i - 0.5f)) - 1.0f/(4.0f*dS*dS)*sinc(float(i - 0.5f)/2.0f)*sinc(float(i - 0.5f)/2.0f));

        nCnt++;
    }
    
    return ;
}

void Filtering(float *pdDst, float *pdSrc, float *pdFlt, int *pnSizeDct, int nNumView)
{
    int nDctY       = pnSizeDct[0];
    int nDctX       = pnSizeDct[1];
    
    int nSizeDct    = nDctY*nDctX;
    
    int nNumFlt     = 2*nDctX - 1;
    
    int nThreadNum  = 8;
    
    dim3    block;
    dim3    grid;
    
    block.x     = nThreadNum;
    grid.x      = (int)ceil(float(nDctX)/nThreadNum);
    
    block.y     = nThreadNum;
    grid.y      = (int)ceil(float(nDctY)/nThreadNum);
    
    block.z     = nThreadNum;
    grid.z      = (int)ceil(float(nNumView)/nThreadNum);
    
    cudaMemcpy(gpdFlt, pdFlt, sizeof(float)*nNumFlt, cudaMemcpyHostToDevice);
    
    conv<<<grid, block, nNumFlt*sizeof(float)>>>(gpdFltP, gpdP, gpdFlt, nDctY, nDctX, nNumView, nNumFlt);
            
    cudaMemcpy(pdDst, gpdFltP, sizeof(float)*nSizeDct*nNumView, cudaMemcpyDeviceToHost);
    
    return ;
}

__global__ void conv(float *pdDst, float *pdSrc, float *pdKernel, int nMY, int nMX, int nMZ, int nN)
{
    int     nXIter      = blockDim.x*blockIdx.x + threadIdx.x;
    int     nYIter      = blockDim.y*blockIdx.y + threadIdx.y;
    int     nZIter      = blockDim.z*blockIdx.z + threadIdx.z;
    
    if (nXIter >= nMX)  return ;
    if (nYIter >= nMY)  return ;
    if (nZIter >= nMZ)  return ;
    
    int     nIdx        = nMX*nMY*nZIter + nMX*nYIter + nXIter;
    
    extern __shared__ float shpdKernel[];

    for (int nIter = 0; nIter < nN; nIter++) {
        shpdKernel[nIter] = pdKernel[nIter];
    }
    
    __syncthreads();
    
    int     nK          = 0;
            
    int     nPreInPos   = 0;
    int     nPostInPos  = 0;
    
    int     nPreOutPos  = 0;
    int     nPostOutPos = 0;
    
    int     nSize       = nMX + nN - 1;
    
    int     nMargin     = int((nSize - nMX)/2) + (nN + 1)%2;

    nPreOutPos  = nMargin;
    nPostOutPos = nMargin + nMX;

    ///////////////////////////////////////////////////////////////////////
    
    nK              = nXIter + nPreOutPos;
    
    if (nK >= nPostOutPos)   return;

    nPreInPos       = 0 > (nK + 1 - nN) ? 0 : (nK + 1 - nN);
    nPostInPos      = nK + 1 < nMX ? nK + 1 : nMX;
    
    float   dSum    = 0;

    for(int nJ = nPreInPos; nJ < nPostInPos; nJ++) {
        dSum	+= pdSrc[nMX*nMY*nZIter + nMX*nYIter + nJ]*shpdKernel[nK - nJ];
        
        __syncthreads();
    }

    pdDst[nIdx]	= dSum;

    return ;
}

void CreateMemory(float *pdSrc, int *pnSizeDct, int nNumView)
{
    int nDctY	= pnSizeDct[0];
    int nDctX	= pnSizeDct[1];
    
    int nSizeDct = nDctY*nDctX;
    int nSize   = nSizeDct*nNumView;
    
    int nNumFlt     = 2*nDctX - 1;
    
    cudaMalloc(&gpdP, sizeof(float)*nSize);                 cudaMemset(gpdP, 0, sizeof(float)*nSize);
    cudaMalloc(&gpdFltP, sizeof(float)*nSize);              cudaMemset(gpdFltP, 0, sizeof(float)*nSize);
    
    cudaMemcpy(gpdP, pdSrc, sizeof(float)*nSize, cudaMemcpyHostToDevice);
    
    cudaMalloc(&gpdFlt, sizeof(float)*nNumFlt);             cudaMemset(gpdFlt, 0, sizeof(float)*nNumFlt);
    pdFlt	= (float *) malloc(sizeof(float)*nNumFlt);      memset(pdFlt, 0, sizeof(float)*nNumFlt);

    return ;
};

void DestroyMemory()
{
    cudaFree(gpdP);     gpdP        = 0;
    cudaFree(gpdFltP);	gpdFltP     = 0;
    
    cudaFree(gpdFlt);	gpdFlt      = 0;
    free(pdFlt);        pdFlt       = 0;
    
    return ;
}
