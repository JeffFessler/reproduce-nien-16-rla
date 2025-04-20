#include "mex.h"
#include <omp.h>
#include <iostream>
#include <cstring>
#include <math.h>

/***** input *****/
// image
#define _i_z       prhs[0]
// regularizer
#define _i_kappa   prhs[3]
#define _i_offsets prhs[4]
#define _i_beta    prhs[5]
#define _i_pot     prhs[6]
#define _i_delta   prhs[7]
#define _i_nthread prhs[9]
// prox average
#define _i_eta     prhs[1]
#define _i_dinv    prhs[2]
#define _i_K       prhs[8]

/***** output *****/
#define _o_coef    plhs[0]

#define BUFFER_LENGTH 127

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define SIGN(a) (((a)>=0)?1:-1)

float shrinkQuad (float z, float reg, float delta) {
    return z/(1+reg);
}

float shrinkHuber (float z, float reg, float delta) {
    return (delta*(1+reg)<fabs(z)) ? z*(1-reg*delta/fabs(z)) : z/(1+reg);
}

float shrinkFairl1 (float z, float reg, float delta) {
    return SIGN(z)*(fabs(z)-(delta+reg)+sqrt(pow(delta+reg-fabs(z),2)+4*delta*fabs(z)))/2;
}

float shrinkAbs (float z, float reg, float delta) {
    return SIGN(z)*MAX(fabs(z)-delta*reg,0);
}

typedef float(*ptr2func)(float, float, float);

ptr2func getPtr (const std::string pot) {
    if (pot=="quad")
        return &shrinkQuad;
    else if (pot=="huber")
        return &shrinkHuber;
    else if (pot=="fair-l1")
        return &shrinkFairl1;
    else if (pot=="abs")
        return &shrinkAbs;
    else
        std::cerr << "undefined potential function!?" << std::endl;
}

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    float *z, *dinv, *kappa;
    double eta, *offsets, *beta, delta, K, nthread;
    z       = (float *)mxGetPr(_i_z);
    eta     = mxGetScalar(_i_eta);
    dinv    = (float *)mxGetPr(_i_dinv);
    kappa   = (float *)mxGetPr(_i_kappa);
    offsets = mxGetPr(_i_offsets);
    beta    = mxGetPr(_i_beta);
    delta   = mxGetScalar(_i_delta);
    K       = mxGetScalar(_i_K);
    nthread = mxGetScalar(_i_nthread);
    
    char *pot = (char *)mxMalloc(sizeof(char)*BUFFER_LENGTH);
    mxGetString(_i_pot, pot, BUFFER_LENGTH);
    
    size_t ndim = mxGetNumberOfDimensions(_i_z);
    const mwSize *dim = mxGetDimensions(_i_z);
    int numel = mxGetNumberOfElements(_i_z);
    int M = mxGetNumberOfElements(_i_offsets);
    
    float *coef;
    _o_coef = mxCreateNumericArray(ndim, dim, mxSINGLE_CLASS, mxREAL);
    coef = (float *)mxGetPr(_o_coef);
    // for (int i=0; i<numel; i++) coef[i] = 0;
    
    float (*shrink)(float, float, float) = NULL;
    shrink = getPtr((std::string)pot);
    
    int i, j;
    float cz, cwc, kap, cz_shrink;
    for (int m=0; m<M; m++) {
        int offset = offsets[m];
        float *alpha = new float [numel+offset] ();
        // two-pass method
        #pragma omp parallel for shared(numel,offset,z,eta,winv,kappa,K,beta,delta,alpha) private(i,j,cz,cwc,kap,cz_shrink) num_threads((int)nthread)
        for (i=0; i<numel-offset; i++) {
            j = i+offset;
            cz = z[i]-z[j];
            cwc = dinv[i]+dinv[j];
            kap = kappa[i]*kappa[j];
            cz_shrink = (*shrink)(cz, (float)(eta*K*beta[m]*kap*cwc), (float)delta);
            alpha[j] = (cwc!=0) ? (cz_shrink-cz)/cwc : 0.0f;
        }
        #pragma omp parallel for shared(offset,alpha,coef) private(i) num_threads((int)nthread)
        for(i=0; i<numel; i++) {
            coef[i] = coef[i]-alpha[i]+alpha[i+offset];
        }
        delete[] alpha;
    }
    
    return;
}
