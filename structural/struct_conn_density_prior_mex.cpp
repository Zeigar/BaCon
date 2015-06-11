#include "mex.h"
#include "blas.h"
#include <iostream>
#include "lapack.h"
#include <math.h>
#include <vector>  
#include <time.h>
#include <random>
#include <algorithm>
#include "matrix.h"

using namespace std;
std::mt19937 gen(time(0));
const double pi = 3.14159265359;

void getPermutation( int permutation[], mwSignedIndex *p, int *u, int *v )
{      
    mwSignedIndex dim = *p;
    
    int i,r;
   
    for (i = 0; i < dim; i++) permutation[i] = i;
    if (*u < dim - 2)
    {
        if (*v == dim - 2)
            swap( permutation[*u], permutation[dim - 1] );
        else
            swap( permutation[*u], permutation[dim - 2] );
    }
    if (*v < dim - 2)
    {
        if (*u == dim - 1)
            swap( permutation[*v], permutation[dim - 2] );
        else
            swap( permutation[*v], permutation[dim - 1] );
    }
    for (i = 0; i < dim - 2; i++)
    {
        r = rand() % ((int)dim - 3); // [1, p-2]
        swap( permutation[i], permutation[r] );       
    }
}
void ind2sub ( const int *edge, mwSignedIndex *p, int &u, int &v )
{
    mwSignedIndex dim = *p;
    int e = *edge;
    u = (e) / dim;
    v = (e) % dim;
}

void edgePermutation( int permutation[] , mwSignedIndex *p)
{
    mwSignedIndex dim = *p, pp = dim*(dim-1)/2;
    int i,r;
    for (i = 0; i < pp; i++) permutation[i] = i;
    for (i = 0; i < pp; i++) 
    {
        r = rand() % ((int)dim - 1);
        swap( permutation[i], permutation[r] );
    }
        
}

void permuteMatrix( const int M[], int Mperm[], const int permutation[], mwSignedIndex *p )
{
    mwSignedIndex dim = *p;
    int i,j;
    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            Mperm[i + j*dim] = M[permutation[i] + permutation[j]*dim];
}
void permuteMatrix( const double M[], double Mperm[], const int permutation[], mwSignedIndex *p )
{
    mwSignedIndex dim = *p;
    int i,j;
    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            Mperm[i + j*dim] = M[permutation[i] + permutation[j]*dim];
}
void unpermuteMatrix( const int M[], int Mperm[], const int permutation[], mwSignedIndex *p )
{
    mwSignedIndex dim = *p;
    int i,j;
    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            Mperm[permutation[i] + permutation[j]*dim] = M[i + j*dim];
}
void unpermuteMatrix( const double M[], double Mperm[], const int permutation[], mwSignedIndex *p )
{
    mwSignedIndex dim = *p;
    int i,j;
    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            Mperm[permutation[i] + permutation[j]*dim] = M[i + j*dim];
}

// copying square matrix A to matrix copyA, for array with one dimension
void copyMatrix( const double A[], double copyA[], mwSignedIndex *pxp )
{
	for( register unsigned short int i = 0, dim = *pxp; i < dim; i++ ) copyA[i] = A[i]; 	
}
// copying square matrix A to matrix copyA, for array with one dimension
void copyMatrix( int A[], int copyA[], mwSignedIndex *pxp )
{
	for( register unsigned short int i = 0, dim = *pxp; i < dim; i++ ) copyA[i] = A[i]; 	
}
	
// Takes square matrix A (p x p) and retrieves square submatrix B (p_sub x p_sub), dictated by vector sub
void subMatrix( double A[], double subA[], int sub[], mwSignedIndex *p_sub, mwSignedIndex *p  )
{
	for( int i = 0, psub = *p_sub, pdim = *p; i < psub; i++ )
		for( register unsigned short int j = 0; j < psub; j++ )
			subA[j * psub + i] = A[sub[j] * pdim + sub[i]]; 
}

void addMatrix( double A[], const double B[], const double C[], mwSignedIndex *p )
{
    mwSignedIndex dim = *p, pxp = dim*dim;
    for (int i = 0; i < pxp; i++) A[i] = B[i] + C[i];    
}
void eye( double I[], mwSignedIndex *p )
{
    mwSignedIndex dim = *p;
    for (int i = 0; i < dim; i++)
    {
        I[i + i*dim] = 1.0;
    }
}
   
////////////////////////////////////////////////////////////////////////////////
//  Multiplies (p_i x p_k) matrix by (p_k x p_j) matrix to give (p_i x p_j) matrix
//  C := A * B
void multiplyMatrix( double A[], double B[], double C[], mwSignedIndex *p_i, mwSignedIndex *p_j, mwSignedIndex *p_k )
{
	double one = 1.0, zero  = 0.0;
	char trans   = 'N';																	
	dgemm( &trans, &trans, p_i, p_j, p_k, &one, A, p_i, B, p_k, &zero, C, p_i );
}




void printMatrix( const int G[], mwSignedIndex *p )
{
    mwSignedIndex dim = *p;
    int i,j;
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < dim; j++) printf("%d  ", G[i + j*dim]);
        printf("\n");
    }
}
void printMatrix( const double M[], mwSignedIndex *p )
{
    mwSignedIndex dim = *p;
    int i,j;
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < dim; j++) printf("%1.4f  ", M[i + j*dim]);
        printf("\n");
    }
}

/* Logarithm of the gamma function.
   Returns NaN for negative arguments, even though log(gamma(x)) may
   actually be defined.
*/
// John Cook
double gammaln(double x)
{
  #define M_lnSqrt2PI 0.91893853320467274178
  static double gamma_series[] = {
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5
  };
  int i;
  double denom, x1, series;
  if(x < 0) return mxGetNaN();
  if(x == 0) return mxGetInf();
  if(mxIsInf(x)) return x;
  /* Lanczos method */
  denom = x+1;
  x1 = x + 5.5;
  series = 1.000000000190015;
  for(i = 0; i < 6; i++) {
    series += gamma_series[i] / denom;
    denom += 1.0;
  }
  return( M_lnSqrt2PI + (x+0.5)*log(x1) - x1 + log(series/x) );
}

double delta_dcm( double N[], double *ap, double *an, int u, int v, int Gprop[], int G[], mwSignedIndex *p)
{
    mwSignedIndex dim = *p;
    double a1 = *ap, a0 = *an, t1, t2, t3, t4, t5, t6, t7, t8;
    int i, uv = u + v*dim, vu = v+u*dim, propedge = Gprop[uv];
//     printf("uv: %d\n", uv);
    
    
    double aprop = propedge*a1 + (1-propedge)*a0, aold = propedge*a0 + (1-propedge)*a1;
    double apropsumk = 0.0, asumk = 0.0, apropsuml = 0.0, asuml = 0.0;
    double sumkprop = 0, sumkold = 0, sumNk = 0, sumlprop = 0, sumlold = 0, sumNl = 0;
    
    for (i = 0; i < dim; i++)
    {
        sumkprop    += Gprop[u + dim*i];
        sumkold     += G[u + dim*i];
        sumNk       += N[u + dim*i];
        sumlprop    += Gprop[v + dim*i];
        sumlold     += G[v + dim*i];
        sumNl       += N[v + dim*i];
    }
//     printf("sum N(k,:) = %2.2f, sum N(l,:) = %2.2f\n", sumNk, sumNl);

    apropsumk = sumkprop * a1 + (dim - sumkprop) * a0;
    asumk = sumkold * a1 + (dim - sumkold) * a0;
    
    
    t1 = gammaln( aprop + N[uv] ) - gammaln ( aold + N[uv] );
    t2 = gammaln( aprop ) - gammaln( aold );
    t3 = gammaln( apropsumk ) - gammaln( asumk );
    t4 = gammaln( apropsumk + sumNk ) - gammaln( asumk + sumNk );
    
    apropsuml = sumlprop * a1 + (dim - sumlprop) * a0;
    asuml = sumlold * a1 + (dim - sumlold) * a0;
    
    t5 = gammaln( aprop + N[vu] ) - gammaln ( aold + N[vu] );
    t6 = gammaln( aprop ) - gammaln( aold );
    t7 = gammaln( apropsuml ) - gammaln( asuml );
    t8 = gammaln( apropsuml + sumNl ) - gammaln( asuml + sumNl );
    
    return (t1 - t2 + t3 - t4 + t5 - t6 + t7 - t8);
}



void MCMC( int G0[], double N[], double Gsamples[], mwSignedIndex *p, double *ap, double *an, double a, double b, int *nsamples )
{
    
    mwSignedIndex dim = *p, pxp = dim*dim, t = *nsamples, m = dim*(dim-1)/2;
    int i, j, e, s, u, v;
    double alpha, logprior;
    
    vector<int> G( pxp ), Gprop ( pxp ), Gperm( pxp ), edgeperm( m );  
    copyMatrix( &G0[0], &G[0], &pxp );
    
    //vector<double> D( pxp ), Dcopy( pxp ), Dinv( pxp ), Dperm( pxp ), Dinvperm( pxp ), U( pxp ), Ucopy( pxp ), Uinv( pxp ), Uperm( pxp ), Kperm( pxp ), K0perm( pxp );
    
    // ---- find linear indices of upper triangle
    e = 0;
    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++)
            if (j > i)
            {
                edgeperm[e] = i+j*dim;
                e++;
            }
    
    uniform_real_distribution<double> toss(0, 1);
    
    // ---- sample P( G, K | X )
    for (s = 0; s < t; s++) 
    {
        // ---- shuffle list over edges
        random_shuffle( &edgeperm[0], &edgeperm[m] ); 
        
        // ---- iterate over edges
        for (e = 0; e < m; e++) 
        {         
            // ---- determine (u,v)            
            ind2sub( &edgeperm[e], p, u, v );  
            
            // ---- Propose a graph G' by flipping one edge
            copyMatrix( &G[0], &Gprop[0], &pxp );
            Gprop[u + dim*v] = 1 - G[u + dim*v];
            Gprop[v + dim*u] = 1 - G[v + dim*u];            
            
            alpha = delta_dcm( &N[0], ap, an, u, v, &Gprop[0], &G[0] , &dim );
            
            // ---- Compute acceptance probability
            if ( Gprop[u + dim*v] ) // additional edge
                logprior = log( a / b );
            else // removed edge
                logprior = log( b / a );
            
            // ---- determine acceptance
            if (  toss( gen ) < min( 1.0, exp( alpha + logprior ) ) ) 
            {
                // ---- accepted, G <- G'
                copyMatrix( &Gprop[0], &G[0], &pxp );
            } 
                        
            
        } // end of full sweep
         
        
        // ---- Store samples in MEX array
        for (i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
            {
                if (i==j) Gsamples[s*pxp + i + j*dim] = 1;
                else  Gsamples[s*pxp + i + j*dim] = G[i + j*dim];
            }
    }
}

 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    
    double *Np, *Gp, *Gsamples, ap, an, a, b;  
    int nsamples;
    size_t nnodes;      
    
    // ---- initialize RNG once
    srand(time(NULL));

    Gp = mxGetPr(prhs[0]);                      /* initial graph, matlab format */
    Np = mxGetPr(prhs[1]);                       /* streamline count */
    ap = mxGetScalar(prhs[2]);                  /* a+ */
    an = mxGetScalar(prhs[3]);                  /* a- */
    a = mxGetScalar(prhs[4]);                   /* prior alpha */
    b = mxGetScalar(prhs[5]);                   /* prior beta */
    nsamples = mxGetScalar(prhs[6]);            /* number of mcmc samples */        
    nnodes = mxGetM(prhs[0]);     
    
    if (nnodes != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
    }
    mwSignedIndex p = (mwSignedIndex)nnodes;
    vector<int> G0( p*p );                      /* initial graph, C++  */
//     vector<int> N(p*p);
    
    
    for (int i = 0; i < p; i++) 
        for (int j = 0; j < p; j++)
        {
            if (Gp[i + j*p] != 0.0 && i!=j) {
                G0[i + j*p] = 1; 
//                 N[i + j*p] = Np[i + j*p];
            }
            else {
                G0[i + j*p] = 0;
//                 N[i + j*p] = 0;
            }
        }

   
    /* Output variables p x p x n */
    mwSize ndims = 3;
    mwSize dims[] = {p,p,nsamples};
    plhs[0] = mxCreateNumericArray(ndims, dims, mxDOUBLE_CLASS, mxREAL);
    Gsamples = mxGetPr(plhs[0]);
    
    // ---- start MCMC algorithm
//     printMatrix( &Np[0], &p );
    MCMC(&G0[0], &Np[0], &Gsamples[0], &p, &ap, &an, a, b, &nsamples);
}
