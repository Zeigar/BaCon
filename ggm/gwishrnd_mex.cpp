#include "mex.h"
#include "blas.h"
#include <iostream>
#include "lapack.h"
#include <math.h>
#include <vector>  
#include <time.h>
#include <random>

using namespace std;
std::mt19937 gen(time(0));

// copying square matrix A to matrix copyA, for arrary with one dimention
void copyMatrix( const double A[], double copyA[], mwSignedIndex *pxp )
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
 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *K, *Sinv, *G;  
    int n;
    size_t p;      /* matrix dimensions */
    

    G = mxGetPr(prhs[0]); /* first input matrix */
    Sinv = mxGetPr(prhs[1]); /* third input matrix */
    n = mxGetScalar(prhs[2]); 
    
    p = mxGetM(prhs[0]);  
    

    if (p != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
    }
    vector<int> G0( p*p );
    
    srand(time(0));
    
    for (int i = 0; i < p; i++) 
        for (int j = 0; j < p; j++)
        {
            if (G[i + j*p] != 0.0 && i!=j)
                G0[i + j*p] = 1; 
            else
                G0[i + j*p] = 0;
        }

    /* create output matrix K */
    plhs[0] = mxCreateDoubleMatrix(p, p, mxREAL);
    
    K = mxGetPr(plhs[0]);
    
    mwSignedIndex numvars = (mwSignedIndex)p;
    gwishrand(&G0[0], Sinv, K, &n, &numvars);
    
   
}
