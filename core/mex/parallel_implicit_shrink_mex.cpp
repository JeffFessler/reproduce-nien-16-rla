#include "mex.h"
#include <omp.h>
#include <iostream>
#include <cstring>
#include <math.h>

/***** input *****/
// image
#define _i_y       prhs[0]
#define _i_winv    prhs[1]
// regularizer
#define _i_kappa   prhs[2]
#define _i_offsets prhs[3]
#define _i_beta    prhs[4]
#define _i_pot     prhs[5]
#define _i_delta   prhs[6]
#define _i_nthread prhs[7]
// prox average
#define _i_N       prhs[7]

/***** output *****/
#define _o_coef   plhs[0]

#define BUFFER_LENGTH 127
//#define NUMBER_OF_THREADS 12

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define SIGN(a) (((a)>=0)?1:-1)

float shrinkQuad (float z, float reg, double delta) {
    return z/(1+reg);
}

float shrinkHuber (float z, float reg, double delta) {
    return (delta*(1+reg)<fabs(z)) ? z*(1-reg*delta/fabs(z)) : z/(1+reg);
}

float shrinkFairl1 (float z, float reg, double delta) {
    return SIGN(z)*(fabs(z)-(delta+reg)+sqrt(pow(delta+reg-fabs(z),2)+4*delta*fabs(z)))/2;
}

float shrinkAbs (float z, float reg, double delta) {
    return SIGN(z)*MAX(fabs(z)-delta*reg,0);
}

float shrinkUser (float z, float reg, double delta) {
    float a = 1;
    float gain = ((a*fabs(z)-1)+sqrt(pow(a*fabs(z)-1,2)-4*a*(delta*reg-fabs(z))))/2/a;
    return (fabs(z)>delta*reg) ? SIGN(z)*gain : 0;
}

typedef float(*ptr2func)(float, float, double);

ptr2func getPtr (const std::string pot) {
    if (pot=="quad")
        return &shrinkQuad;
    else if (pot=="huber")
        return &shrinkHuber;
    else if (pot=="fair-l1")
        return &shrinkFairl1;
    else if (pot=="abs")
        return &shrinkAbs;
    else if (pot=="user")
        return &shrinkUser;
    else
        std::cerr << "undefined potential function!?" << std::endl;
}

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    float *y, *winv, *kappa;
    double *offsets, *beta, delta, N, nthread;
    y       = (float *)mxGetPr(_i_y);
    winv    = (float *)mxGetPr(_i_winv);
    kappa   = (float *)mxGetPr(_i_kappa);
    offsets = mxGetPr(_i_offsets);
    beta    = mxGetPr(_i_beta);
    delta   = mxGetScalar(_i_delta);
    N       = mxGetScalar(_i_N);
    nthread = mxGetScalar(_i_nthread);
    
    char *pot = (char *)mxMalloc(sizeof(char)*BUFFER_LENGTH);
    mxGetString(_i_pot, pot, BUFFER_LENGTH);
    
    size_t ndim = mxGetNumberOfDimensions(_i_y);
    const mwSize *dim = mxGetDimensions(_i_y);
    int numel = mxGetNumberOfElements(_i_y);
    int M = mxGetNumberOfElements(_i_offsets);
    
    float *coef;
    _o_coef = mxCreateNumericArray(ndim, dim, mxSINGLE_CLASS, mxREAL);
    coef = (float *)mxGetPr(_o_coef);
    
    float (*shrink)(float, float, double) = NULL;
    shrink = getPtr((std::string)pot);
    
    int i, j;
    float cy, cwc, kap, reg, cy_shrink;
    for (int m=0; m<M; m++) {
        int offset = offsets[m];
        float *alpha = new float [numel+offset] ();
        // two-pass method
        #pragma omp parallel for shared(numel,offset,y,winv,kappa,N,beta,delta,alpha) private(i,j,cy,cwc,kap,cy_shrink) num_threads((int)nthread)
        for (i=0; i<numel-offset; i++) {
            j = i+offset;
            cy = y[i]-y[j];
            cwc = winv[i]+winv[j];
            kap = kappa[i]*kappa[j];
            cy_shrink = (*shrink)(cy, N*beta[m]*kap*cwc, delta);
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
