#include "mex.h"
#include "cuda.h"

#define     Y   0
#define     X   1
#define     Z   2

///////////////////////////////////////////////////////////////////////////

            float	*gpupdDst   = 0;
            float   *gpupdSrc   = 0;

///////////////////////////////////////////////////////////////////////////

            void    CreateMemory(float *pdSrc, int *pnSize);
            void    DestroyMemory();
            
///////////////////////////////////////////////////////////////////////////
            
            void    RunDy(float *pdDst, float *pdSrc, int *pnSizeSrc);
__global__  void    Dy(float *pdDst, float *pdSrc, int nY, int nX, int nZ);

///////////////////////////////////////////////////////////////////////////

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float   *pdDst      = (float *) mxGetData(prhs[0]);
    float   *pdSrc      = (float *) mxGetData(prhs[1]);
    int     *pnSizeSrc  = (int *)   mxGetData(prhs[2]);
    
    RunDy(pdDst, pdSrc, pnSizeSrc);
    
    return ;
}

void RunDy(float *pdDst, float *pdSrc, int *pnSizeSrc)
{
    
    CreateMemory(pdSrc, pnSizeSrc);
    
    ///////////////////////////////////////////////////////////////////////
    
    int     nY          = pnSizeSrc[Y];
    int     nX          = pnSizeSrc[X];
    int     nZ          = pnSizeSrc[Z];
    
    int     nThreadNum      = 8;
    
    dim3    nBlockNum(nThreadNum, nThreadNum, nThreadNum);
    dim3    nGridNum(ceil(nY/(float)nThreadNum), ceil(nX/(float)nThreadNum), ceil(nZ/(float)nThreadNum));
    
    Dy<<<nGridNum, nBlockNum>>>(gpupdDst, gpupdSrc, nY, nX, nZ);
    
    cudaMemcpy(pdDst, gpupdDst, sizeof(float)*nY*nX*nZ, cudaMemcpyDeviceToHost);
    
    ///////////////////////////////////////////////////////////////////////
    
    DestroyMemory();
    
    return ;
}

__global__  void    Dy(float *pdDst, float *pdSrc, int nY, int nX, int nZ)
{
    int     nIdxY       = blockDim.y*blockIdx.y + threadIdx.y;
    int     nIdxX       = blockDim.x*blockIdx.x + threadIdx.x;
    int     nIdxZ       = blockDim.z*blockIdx.z + threadIdx.z;
    
    if(nIdxY >= nY)     return ;
    if(nIdxX >= nX)     return ;
    if(nIdxZ >= nZ)     return ;
    
    int     nIdxYpost   = nIdxY + 1 < nY ? nIdxY + 1: 0;
    
    int     nIdx        = nY*nX*nIdxZ	+ nY*nIdxX 	+ nIdxY;
    int     nIdxpost	= nY*nX*nIdxZ	+ nY*nIdxX	+ nIdxYpost;
    
    
    pdDst[nIdx]         = + pdSrc[nIdxpost] - pdSrc[nIdx];
    
    return ;
}

void    CreateMemory(float *pdSrc, int *pnSize)
{
    int nY  = pnSize[Y];
    int nX  = pnSize[X];
    int nZ  = pnSize[Z];
    
    cudaMalloc(&gpupdDst, sizeof(float)*nX*nY*nZ);
    cudaMemset(gpupdDst, 0, sizeof(float)*nX*nY*nZ);
    
    cudaMalloc(&gpupdSrc, sizeof(float)*nX*nY*nZ);
    cudaMemset(gpupdSrc, 0, sizeof(float)*nX*nY*nZ);
    cudaMemcpy(gpupdSrc, pdSrc, sizeof(float)*nY*nX*nZ, cudaMemcpyHostToDevice);
            
    return ;
}

void    DestroyMemory()
{
    cudaFree(gpupdSrc);     gpupdSrc    = 0;
    cudaFree(gpupdDst);     gpupdDst    = 0;
    
    return ;
}