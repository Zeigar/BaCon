#include "mex.h"
#include "blas.h"
#include <iostream>
#include "lapack.h"
#include <math.h>
#include <vector>  
#include <time.h>
#include <random>
#include <algorithm>

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

// inverse function for symmetric positive-definite matrices (p x p)
// ******** WARNING: this function change matrix A **************************
void inverse( double A[], double A_inv[], mwSignedIndex *p )
{
	mwSignedIndex info, dim = *p;
	char uplo = 'U';

	// creating an identity matrix
	for( unsigned short int i = 0; i < dim; i++ )
		for(register unsigned short int j = 0; j < dim; j++ )
			A_inv[j * dim + i] = (i == j);
	
	// LAPACK function: computes solution to A x X = B, where A is symmetric positive definite matrix
	dposv( &uplo, &dim, &dim, A, &dim, A_inv, &dim, &info );
}

// Cholesky decomposition function for symmetric positive-definite matrices (p x p)
// ******** WARNING: lower triangle is not set to zero! **************************
void cholesky (double A[], double R[], mwSignedIndex *p)
{
    mwSignedIndex info, dim = *p, pxp = dim*dim;
    char T = 'U';
    copyMatrix( A, R, &pxp );    
    dpotrf( &T, &dim, &R[0], &dim, &info );
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


// sampling from Wishart distribution
// Ti = chol( solve( D ) )
void wishrand( double Ti[], double K[], int *b, mwSignedIndex *p )
{
	mwSignedIndex dim = *p;
    int pxp = dim * dim, bK = *b;
    
	vector<double> psi( pxp ); 
    
    
    double mean = 0.0;
    double stdev  = 1.0;
    std::normal_distribution<double> normrand(mean, stdev);

	// ---- Sample values in Psi matrix ---
	for( unsigned short int i = 0; i < dim; i++ )
		for( register unsigned short int j = 0; j < dim; j++ ){
            std::chi_squared_distribution<double> chisq(bK + dim - i - 1);   
			psi[j * dim + i] = (i < j) ? normrand(gen) : ( (i > j) ? 0.0 : sqrt( chisq( gen ) ) );
        }
	// ------------------------------------
    // C = psi %*% Ti 
    vector<double> C( pxp ); 
	multiplyMatrix( &psi[0], Ti, &C[0], &dim, &dim, &dim );


	// K = t(C) %*% C 
	double alpha = 1.0, beta  = 0.0;
	char transA  = 'T', transB  = 'N';
	// LAPACK function to compute  C := alpha * A * B + beta * C																				
	dgemm( &transA, &transB, &dim, &dim, &dim, &alpha, &C[0], &dim, &C[0], &dim, &beta, K, &dim );
}

// A is adjacency matrix which has zero in its diagonal
// threshold = 1e-8
void gwishrand( const int G[], const double S[], double K[], int *b, mwSignedIndex *p )
{
	register int k;
    mwSignedIndex dim = *p, a, pxp = dim * dim, one = 1;
    int i,j, l;	
	double temp;
    
    // ---- chol(S)
    vector<double> Ti( pxp ), copyS( pxp ), invS( pxp );
    copyMatrix( &S[0], &copyS[0], &pxp );   
    cholesky( &copyS[0], &Ti[0], p ); 
    // ---- Set lower triangle to zero 
    for (i=0; i<dim; i++){
        for (j=i+1; j<dim; j++){
             Ti[j + i*dim] = 0;
        }
    }
    
    // ---- K ~ Wish(b,chol(S))
	wishrand( &Ti[0], K, b, &dim ); 
	
	vector<double> Sigma( pxp ); 
	inverse( K, &Sigma[0], &dim );
	
	// copying  matrix sigma to matrix W	
	vector<double> W( Sigma ); 

	vector<double> W_last( pxp ); 
	vector<double> beta_star( dim ); 
	vector<double> ww( dim ); 

	double difference = 1.0;	
	while ( difference > 1e-8 )
	{
		// copying  matrix W to matrix W_last	
		copyMatrix( &W[0], &W_last[0], &pxp ); 	
		
		for( j = 0; j < dim; j++ )
		{
			// Count  size of note
			a = 0;
			for( k = 0; k < dim; k++ ) a += G[k * dim + j];

			if( a > 0 )
			{
				// Record size of node and initialize zero in beta_star for next steps
				vector<double> Sigma_N_j( a );
				vector<int> N_j( a );
				l = 0;
				for( k = 0; k < dim; k++ )
				{
					if( G[k * dim + j] )
					{
						Sigma_N_j[l] = Sigma[j * dim + k]; // Sigma_N_j[k] = Sigma[j * dim + N_j[k]];
						N_j[l++]     = k;
					}
					
					beta_star[k] = 0.0; // for( k = 0; k < *p; k++ ) beta_star[k] = 0.0;
				}
				// -------------------------------------------------------------
				
				vector<double> W_N_j( a * a );
				subMatrix( &W[0], &W_N_j[0], &N_j[0], &a, &dim );
				
				vector<double> W_N_j_inv( a * a );
				inverse( &W_N_j[0], &W_N_j_inv[0], &a );
				
				vector<double> beta_star_hat_j( a );   
				multiplyMatrix( &W_N_j_inv[0], &Sigma_N_j[0], &beta_star_hat_j[0], &a, &one, &a );
				
				for( k = 0; k < a; k++ ) beta_star[N_j[k]] = beta_star_hat_j[k];
				
				multiplyMatrix( &W[0], &beta_star[0], &ww[0], &dim, &one, &dim );
				
				for( k = 0; k < j; k++ )
				{
					W[k * dim + j] = ww[k];
					W[j * dim + k] = ww[k];
				}
				
				for( k = j + 1; k < dim; k++ )
				{
					W[k * dim + j] = ww[k];
					W[j * dim + k] = ww[k];
				}
			} 
			else 
			{
				for( k = 0; k < j; k++ )
				{
					W[k * dim + j] = 0.0;
					W[j * dim + k] = 0.0;
				}
				
				for( k = j + 1; k < dim; k++ )
				{
					W[k * dim + j] = 0.0;
					W[j * dim + k] = 0.0;
				}
			} 
		}

		difference = fabs( W[0] - W_last[0] );
		for( k = 1; k < pxp; k++ )
		{
			temp = fabs( W[k] - W_last[k] );
			if( temp > difference ) difference = temp; 
		}		
	}

	inverse( &W[0], K, &dim );
}

double logncbf( const double K[], const double S[], mwSignedIndex *p )
{ // DEBUG: triple checked with Matlab implementation, always identical results
    mwSignedIndex dim = *p, pxp = dim*dim;
    vector<double> Kcopy( pxp ), R( pxp );
    copyMatrix( &K[0], &Kcopy[0], &pxp );
    cholesky( &Kcopy[0], &R[0], p );
    
    double mu, R0, N, tmp = 0.0, Spp = S[dim-1 + dim*(dim-1)], Rp1p1 = R[dim-2 + dim*(dim-2)];
    mu = Rp1p1 * S[dim-2 + dim*(dim-1)] / Spp;
    R0 = -1.0 / Rp1p1;
    for (int i = 0; i < dim - 2; i++)
        tmp += R[i + dim*(dim-2)] * R[i + dim*(dim-1)];
    R0 *= tmp;
    
    N = log(  Rp1p1 ) + 0.5*log(2*pi / Spp) + 0.5*Spp*(R0 + mu)*(R0 + mu);  
    return N;
}


void BGGM( int G0[], double S[], double Gsamples[], double Ksamples[], mwSignedIndex *p, int *n, int *nsamples )
{
    
    mwSignedIndex dim = *p, pxp = dim*dim, t = *nsamples, m = dim*(dim-1)/2;
    int i, j, e, s, u, v, d=3, dn=d + *n;    
    double alpha, logprior, a=1.0, b=1.0;
    mxArray *Kptr = mxCreateDoubleMatrix( dim, dim, mxREAL );    
    double *K = mxGetPr(Kptr);     
    
    vector<int> G( pxp ), Gprop ( pxp ), Gperm( pxp ), permutation( dim ), edgeperm( m );  
    copyMatrix( &G0[0], &G[0], &pxp );
    
    vector<double> D( pxp ), Dcopy( pxp ), Dinv( pxp ), Dperm( pxp ), Dinvperm( pxp ), U( pxp ), Ucopy( pxp ), Uinv( pxp ), Uperm( pxp ), Kperm( pxp ), K0perm( pxp );
    eye( &D[0], p ); 
    copyMatrix( &D[0], &Dcopy[0], p );
    inverse( &Dcopy[0], &Dinv[0], p );
    addMatrix( &U[0], &S[0], &D[0], p );    
    copyMatrix( &U[0], &Ucopy[0], &pxp );
    inverse( &Ucopy[0], &Uinv[0], p );
    
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
            
            // ---- Draw K | G ~ G-Wishart
            gwishrand( &G[0], &Uinv[0], K, &dn, p ); 
            
            // ---- get random permutation of 1:p s.t. (p-1,p) = (u,v)
            getPermutation( &permutation[0], p, &u, &v ); 
            
            // ---- Permute entire system
            permuteMatrix( &G[0],       &Gperm[0],      &permutation[0],    p );
            permuteMatrix( &K[0],       &Kperm[0],      &permutation[0],    p );
            permuteMatrix( &U[0],       &Uperm[0],      &permutation[0],    p );
            permuteMatrix( &D[0],       &Dperm[0],      &permutation[0],    p );
            permuteMatrix( &Dinv[0],    &Dinvperm[0],   &permutation[0],    p );
            
            // ---- Propose a graph G' by flipping one edge
            copyMatrix( &Gperm[0], &Gprop[0], &pxp );
            Gprop[dim-2 + dim*(dim-1)] = 1 - Gperm[dim-2 + dim*(dim-1)];
            Gprop[dim-1 + dim*(dim-2)] = 1 - Gperm[dim-1 + dim*(dim-2)];            
            
            // ---- Draw K0 | G' ~ GWish(d,D)
            gwishrand( &Gprop[0], &Dinvperm[0], &K0perm[0], &d, p );
            
            // ---- Compute acceptance probability
            if ( Gprop[dim-2 + dim*(dim-1)] ) // additional edge
            {
                alpha = logncbf( &Kperm[0], &Uperm[0], p ) - logncbf( &K0perm[0], &Dperm[0], p );
                logprior = log( a / b );
            }
            else // removed edge
            {
                alpha = logncbf( &K0perm[0], &Dperm[0], p ) - logncbf( &Kperm[0], &Uperm[0], p );
                logprior = log( b / a );
            }
            
            // ---- determine acceptance
            if (  toss( gen ) < min( 1.0, exp( alpha + logprior ) ) ) 
            {
                // ---- accepted, G <- G'
                copyMatrix( &Gprop[0], &Gperm[0], &pxp );
            } 
            unpermuteMatrix( &Gperm[0], &G[0], &permutation[0], p );
                        
            
        } // end of full sweep
         
        // ---- Draw K | G ~ GWish(d+n,D+S)
        gwishrand(&G[0], &Uinv[0], K, &dn, p);
        
        
        // ---- Store samples in MEX array
        for (i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
            {
                if (i==j) Gsamples[s*pxp + i + j*dim] = 1;
                else  Gsamples[s*pxp + i + j*dim] = G[i + j*dim];
                Ksamples[s*pxp + i + j*dim] = K[i + j*dim];
            }
    }
    //delete[] &D[0], &Dinv[0], &Dperm[0], &Dinvperm[0], &U[0], &Uinv[0], &Uperm[0], &Kperm[0], &K0perm[0];
}

 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    
    double *S, *Gp, *Gsamples, *Ksamples;  
    int n, nsamples;
    size_t nnodes;      
    
    // ---- initialize RNG once
    srand(time(NULL));

    Gp = mxGetPr(prhs[0]);                      /* initial graph, matlab format */
    S = mxGetPr(prhs[1]);                       /* empirical covariance/correlation (z-scored data) */
    n = mxGetScalar(prhs[2]);                   /* number of data samples */
    nsamples = mxGetScalar(prhs[3]);            /* number of ggm samples */        
    nnodes = mxGetM(prhs[0]);     
    
    if (nnodes != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
    }
    mwSignedIndex p = (mwSignedIndex)nnodes;
    vector<int> G0( p*p );                      /* initial graph, C++  */
    
    
    for (int i = 0; i < p; i++) 
        for (int j = 0; j < p; j++)
        {
            if (Gp[i + j*p] != 0.0 && i!=j)
                G0[i + j*p] = 1; 
            else
                G0[i + j*p] = 0;
        }

   
    /* Output variables p x p x n */
    mwSize ndims = 3;
    mwSize dims[] = {p,p,nsamples};
    plhs[0] = mxCreateNumericArray(ndims, dims, mxDOUBLE_CLASS, mxREAL);
    Gsamples = mxGetPr(plhs[0]);
    plhs[1] = mxCreateNumericArray(ndims, dims, mxDOUBLE_CLASS, mxREAL);
    Ksamples = mxGetPr(plhs[1]);
    
    // ---- start GGM algorithm
    BGGM(&G0[0], &S[0], &Gsamples[0], &Ksamples[0], &p, &n, &nsamples);
}
