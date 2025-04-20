#include "mex.h"
#include <omp.h>
#include <math.h>

/***** input *****/
// image
#define _i_y       prhs[0]
#define _i_winv    prhs[1]
// regularizer
#define _i_offsets prhs[2]
#define _i_beta    prhs[3]
#define _i_nthread prhs[4]
// prox average
#define _i_N       prhs[5]

/***** output *****/
#define _o_coef   plhs[0]

#define MAX(a,b) (((a)>(b))?(a):(b))
#define SIGN(a) (((a)>=0)?1:-1)

double shrinkAbs (double z, double reg) {
    return SIGN(z)*MAX(fabs(z)-reg,0);
}

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *y, *winv;
    double *offsets, *beta, N, nthread;
    y       = (double *)mxGetPr(_i_y);
    winv    = (double *)mxGetPr(_i_winv);
    offsets = mxGetPr(_i_offsets);
    beta    = mxGetPr(_i_beta);
    N       = mxGetScalar(_i_N);
    nthread = mxGetScalar(_i_nthread);
    
    size_t ndim = mxGetNumberOfDimensions(_i_y);
    const mwSize *dim = mxGetDimensions(_i_y);
    int numel = mxGetNumberOfElements(_i_y);
    int M = mxGetNumberOfElements(_i_offsets);
    
    double *coef;
    _o_coef = mxCreateNumericArray(ndim, dim, mxDOUBLE_CLASS, mxREAL);
    coef = (double *)mxGetPr(_o_coef);
    
    int i, j;
    double cy, cwc, cy_shrink;
    for (int m=0; m<M; m++) {
        int offset = offsets[m];
        double *alpha = new double [numel+offset] ();
        // two-pass method
        #pragma omp parallel for shared(numel,offset,y,winv,N,beta,alpha) private(i,j,cy,cwc,cy_shrink) num_threads((int)nthread)
        for (i=0; i<numel-offset; i++) {
            j = i+offset;
            cy = y[i]-y[j];
            cwc = winv[i]+winv[j];
            cy_shrink = shrinkAbs(cy, N*beta[m]*cwc);
            alpha[j] = (cwc!=0) ? (cy_shrink-cy)/cwc : 0;
        }
        #pragma omp parallel for shared(offset,alpha,coef) private(i) num_threads((int)nthread)
        for(i=0; i<numel; i++) {
            coef[i] = coef[i]-alpha[i]+alpha[i+offset];
        }
        delete[] alpha;
    }
    
    return;
}
