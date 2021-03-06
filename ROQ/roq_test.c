#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_randist.h>

#define TSSIZE 10
#define WL 10000

typedef struct tagInterpolant{
  gsl_matrix *Bs;
  int *nodes;
}Interpolant;

/* function to create an orthonormal basis set from a training set of models */
double *create_basis(double weight, double tolerance, gsl_matrix *TS, size_t *nbases);

/* function to create the empirical interpolant */
//void create_interpolant(gsl_matrix *RB, gsl_matrix **Bs, int **idxarray);
Interpolant *create_interpolant(gsl_matrix *RB);

/* define function to project model vector onto the training set of models */
void project_onto_basis(double dt, gsl_matrix *RB, gsl_matrix *TS, gsl_matrix *projections, gsl_matrix *projection_coefficients, int idx);

/* dot product of two vectors, but with one multiplied by the weighting factor */
double weighted_dot_product(double weight, gsl_vector *a, gsl_vector *b);

/* normalise a vector */
void normalise(double weight, gsl_vector *a);

/* get the B_matrix */
gsl_matrix *B_matrix(gsl_matrix *V, gsl_matrix *RB);

void print_shape(gsl_matrix *a);

/* create ROQ weights for interpolant to calculate the data dot model terms */
gsl_vector *create_data_model_weights(gsl_matrix *B, double *data);

/* calculate ROQ version of the data model dot product (where the model
   vector is the model just computed at the interpolant points) */
double roq_data_dot_model(gsl_vector *weights, double *model);

/* create ROQ weights for interpolant to calculate the model dot model terms */
gsl_matrix *create_model_model_weights(gsl_matrix *B);

/* calculate ROQ version of the model model dot product (where the model
   vector is the model just computed at the interpolant points) */
double roq_model_dot_model(gsl_matrix *weights, double *model);


double *create_basis(double weight, double tolerance, gsl_matrix *TS, size_t *nbases){
  double *RB = NULL;

  gsl_matrix *projections; /* projections of the basis onto the training set */
  gsl_matrix *residual;
  gsl_matrix *projection_coeffs;
  gsl_vector *projection_errors;
  gsl_vector_view resrow;

  double sigma = 1.;
  size_t dlength = TS->size2, nts = TS->size1;
  size_t mindex = 0, k=0;
  int idx = 0;

  /* allocate reduced basis (initially just one model vector in length).
     Here we have RB just as a double array, rather than a gsl_matrix,
     as there is no realloc functions for GSL matrices. When wanting to
     use RB as a GSL matrix we will have to use matrix views. */
  RB = calloc(dlength, sizeof(double));  

  gsl_matrix_view RBview = gsl_matrix_view_array(RB, 1, dlength);
  gsl_vector *firstrow = gsl_vector_calloc(dlength);
  gsl_matrix_get_row(firstrow, TS, 0);
  gsl_matrix_set_row(&RBview.matrix, 0, firstrow);
  gsl_vector_free(firstrow);

  projection_errors = gsl_vector_calloc(dlength);
  residual = gsl_matrix_calloc(nts, dlength);
  projections = gsl_matrix_calloc(nts, dlength);
  projection_coeffs = gsl_matrix_calloc(nts, nts);

  /* create reduced basis set using greedy binning Algorithm 1 of http://arxiv.org/abs/1308.3565 */
  while ( sigma >= tolerance ){
    if ( idx > nts-1 ){
      fprintf(stderr, "Not enough training models (%zu) to produce orthonormal basis given the tolerance of %le\n", nts, tolerance);
      return NULL;
    }

    project_onto_basis(weight, &RBview.matrix, TS, projections, projection_coeffs, idx);

    gsl_matrix_memcpy(residual, TS); /* copy training set into residual */

    /* get residuals by subtracting projections from training set */
    gsl_matrix_sub(residual, projections);

    /* get projection errors */
    for( k=0; k < nts; k++ ){
      double err;

      resrow = gsl_matrix_row(residual, k);
      err = weighted_dot_product(weight, &resrow.vector, &resrow.vector);

      gsl_vector_set(projection_errors, k, err);
    }

    sigma = fabs(gsl_vector_max(projection_errors));

    if ( sigma > 1e-5 ){ fprintf(stderr, "%.12lf\t%d\n", sigma, idx); }
    else { fprintf(stderr, "%.12le\t%d\n", sigma, idx); }

    if ( sigma < tolerance ) { break; }

    /* get index of training set with the largest projection errors */
    mindex = gsl_vector_max_index( projection_errors );

    gsl_vector *next_basis = gsl_vector_calloc(dlength);
    gsl_vector_view proj_basis = gsl_matrix_row(projections, mindex);
    gsl_matrix_get_row(next_basis, TS, mindex);
    gsl_vector_sub(next_basis, &proj_basis.vector);

    /* normalise vector */
    normalise(weight, next_basis);

    idx++;

    /* expand reduced basis */
    RB = realloc(RB, sizeof(double)*dlength*(idx+1));

    /* add on next basis */
    RBview = gsl_matrix_view_array(RB, idx+1, dlength);
    gsl_matrix_set_row(&RBview.matrix, idx, next_basis);

    gsl_vector_free(next_basis);
  }

  *nbases = (size_t)idx;

  gsl_matrix_free(projection_coeffs);
  gsl_matrix_free(residual);
  gsl_matrix_free(TS);

  return RB;
}

void project_onto_basis(double dt, gsl_matrix *RB, gsl_matrix *TS, gsl_matrix *projections, gsl_matrix *projection_coefficients, int idx){
  size_t row = 0;
  
  gsl_vector_view basis = gsl_matrix_row(RB, idx);

  for ( row=0; row < TS->size1; row++ ){
    double prod;
    gsl_vector_view proj = gsl_matrix_row(projections, row);
    gsl_vector *basisscale = gsl_vector_calloc(TS->size2);

    gsl_vector_view model = gsl_matrix_row(TS, row); /* get model from training set */

    prod = weighted_dot_product(dt, &basis.vector, &model.vector);
    
    gsl_matrix_set(projection_coefficients, idx, row, prod);
    gsl_vector_memcpy(basisscale, &basis.vector);    
    gsl_vector_scale(basisscale, prod);
    gsl_vector_add(&proj.vector, basisscale);
    gsl_vector_free(basisscale);
  }
}

double weighted_dot_product(double weight, gsl_vector *a, gsl_vector *b){
  double dp;
  gsl_vector *weighted = gsl_vector_calloc(a->size);
  gsl_vector_memcpy(weighted, a);
  
  /* multiply vector by weight */
  gsl_vector_scale(weighted, weight);

  /* get dot product */
  gsl_blas_ddot(weighted, b, &dp);
  
  gsl_vector_free(weighted);

  return dp;
}

void normalise(double weight, gsl_vector *a){
  double norm = gsl_blas_dnrm2(a); /* use GSL normalisation calculation function */

  gsl_vector_scale(a, 1./(norm*sqrt(weight)));
}

gsl_matrix *B_matrix(gsl_matrix *V, gsl_matrix *RB){
  /* get inverse of V */
  size_t n = V->size1;
  gsl_matrix *invV = gsl_matrix_alloc(n, n);
  int signum;
  
  /* use LU decomposition to get inverse */
  gsl_matrix *LU = gsl_matrix_alloc(n, n);
  gsl_matrix_memcpy(LU, V);

  gsl_permutation *p = gsl_permutation_alloc(n);
  gsl_linalg_LU_decomp(LU, p, &signum);
  gsl_linalg_LU_invert(LU, p, invV);
  gsl_permutation_free(p);
  gsl_matrix_free(LU);

  /* get B matrix */
  gsl_matrix_view subRB = gsl_matrix_submatrix(RB, 0, 0, n, RB->size2);
  gsl_matrix *B = gsl_matrix_alloc(n, RB->size2);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, invV, &subRB.matrix, 0., B); 

  gsl_matrix_free(invV);
  return B;
}

void print_shape(gsl_matrix *a){
  fprintf(stderr, "Matrix is %zu x %zu\n", a->size1, a->size2);
}

gsl_vector *create_data_model_weights(gsl_matrix *B, double *data){
  gsl_vector_view dataview = gsl_vector_view_array(data, B->size2);

  /* create weights */
  gsl_vector *weights = gsl_vector_alloc(B->size1);

  gsl_blas_dgemv(CblasNoTrans, 1.0, B, &dataview.vector, 0., weights);

  return weights;
}

double roq_data_dot_model(gsl_vector *weights, double *model){
  double d_dot_m = 0.;
  gsl_vector_view modelview = gsl_vector_view_array(model, weights->size);
  
  gsl_blas_ddot(weights, &modelview.vector, &d_dot_m);

  return d_dot_m;
}

gsl_matrix *create_model_model_weights(gsl_matrix *B){
  gsl_matrix *weights = gsl_matrix_alloc(B->size1, B->size1);
  size_t i=0, j=0;
  double ressum = 0.;

  for ( i=0; i<B->size1; i++ ){
    for ( j=0; j<B->size1; j++ ){
      gsl_vector_view Bi = gsl_matrix_row(B, i);
      gsl_vector_view Bj = gsl_matrix_row(B, j);
      gsl_blas_ddot(&Bi.vector, &Bj.vector, &ressum);
      gsl_matrix_set(weights, i, j, ressum);
    }
  }

  return weights;
}

double roq_model_dot_model(gsl_matrix *weights, double *model){
  gsl_vector *ws = gsl_vector_alloc(weights->size1);
  gsl_vector_view modelview = gsl_vector_view_array(model, weights->size1);
  double m_dot_m = 0.;  

  gsl_blas_dgemv(CblasTrans, 1.0, weights, &modelview.vector, 0., ws);
  gsl_blas_ddot(ws, &modelview.vector, &m_dot_m);

  return m_dot_m;
}

Interpolant *create_interpolant(gsl_matrix *RB){
  /* now find the empirical interopolant and interpolation nodes using Algorithm 2
     of http://arxiv.org/abs/1308.3565 */
  size_t RBsize = RB->size1; /* reduced basis size (no. of reduced bases) */
  size_t dlength = RB->size2; /* length of each base */
  size_t i=1, j=0, k=0;
  double *V = malloc(sizeof(double));
  gsl_matrix_view Vview;
  
  Interpolant *interp = malloc(sizeof(Interpolant));
  
  int idmax = 0, newidx = 0;

  /* get index of maximum absolute value of first basis */
  gsl_vector_view firstbasis = gsl_matrix_row(RB, 0);
  idmax = (int)gsl_blas_idamax(&firstbasis.vector); /* function gets index of maximum absolute value */

  interp->nodes = malloc(RBsize*sizeof(int));
  
  interp->nodes[0] = idmax;

  fprintf(stderr, "first index = %d\n", interp->nodes[0]); 
  
  for ( i=1; i<RBsize; i++ ){
   Vview = gsl_matrix_view_array(V, i, i);

   for ( j=0; j<i; j++ ){
      for ( k=0; k<i; k++ ){
        gsl_matrix_set(&Vview.matrix, k, j, gsl_matrix_get(RB, j, interp->nodes[k]));
      }
    }

    /* get B matrix */
    gsl_matrix *B = B_matrix(&Vview.matrix, RB);

    /* make empirical interpolant of basis */
    gsl_vector *interpolant = gsl_vector_calloc(dlength);
    gsl_vector *subbasis = gsl_vector_calloc(i);
    gsl_vector_view subview = gsl_matrix_row(RB, i);

    for ( k=0; k<i; k++ ){
      gsl_vector_set(subbasis, k, gsl_vector_get(&subview.vector, interp->nodes[k]));
    }

    gsl_blas_dgemv(CblasTrans, 1.0, B, subbasis, 0., interpolant); 

    /* get residuals of interpolant */
    gsl_vector_sub(interpolant, &subview.vector);

    newidx = (int)gsl_blas_idamax(interpolant);

    interp->nodes[i] = newidx;

    fprintf(stderr, "%zu: idx[%d]\n", i, interp->nodes[i]);

    gsl_vector_free(subbasis);
    gsl_matrix_free(B);
    gsl_vector_free(interpolant);

    /* reallocate memory for V */
    V = realloc(V, (i+1)*(i+1)*sizeof(double));
  }

  /* NOTE: the above works and produces identical results to the ipython notebook */

  /* get final B vector with all the indices */
  Vview = gsl_matrix_view_array(V, RBsize, RBsize);
  for( j=0; j<RBsize; j++ ){
    for( k=0; k<RBsize; k++ ){
      gsl_matrix_set(&Vview.matrix, k, j, gsl_matrix_get(RB, j, interp->nodes[k]));
    }
  }

  interp->Bs = B_matrix(&Vview.matrix, RB);

  free(V);  

  return interp;
}

int main(){
  gsl_matrix *TS; /* the training set of waveforms */

  size_t TSsize;  /* the size of the training set (number of waveforms) */
  size_t wl;      /* the length of each waveform */
  size_t k = 0, j = 0, i = 0, nbases = 0;

  double *RB = NULL; /* the reduced basis set */
  Interpolant *interp = NULL;
  
  gsl_vector *times;

  double tolerance = 1e-12, sigma = 1.; /* tolerance for reduced basis generation loop */

  double dt = 60.; /* model time steps */

  TSsize = TSSIZE;
  wl = WL;

  /* allocate memory for training set */
  TS = gsl_matrix_calloc(TSsize, wl);

  double fmin = -0.0001, fmax = 0.0001, f0 = 0., m0 = 0.;
  times = gsl_vector_alloc(wl);

  /* set up training set */
  for ( k=0; k < TSsize; k++ ){
    f0 = fmin + (double)k*(fmax-fmin)/((double)TSsize-1.);

    for ( j=0; j < wl; j++ ){
      double tv = dt*(double)j;
      m0 = sin(2.*M_PI*f0*tv);
      gsl_vector_set(times, j, tv);
      gsl_matrix_set(TS, k, j, m0);
    }

    gsl_vector_view rowview = gsl_matrix_row(TS, k);
    normalise(dt, &rowview.vector);
  }

  if ( (RB = create_basis(dt, tolerance, TS, &nbases)) == NULL){
    fprintf(stderr, "Error... problem producing basis\n");
    return 1;
  }
  
  gsl_matrix_view RBview = gsl_matrix_view_array(RB, nbases, wl);

  fprintf(stderr, "%zu, %zu x %zu\n", nbases, RBview.matrix.size1, RBview.matrix.size2); 

  //FILE *fp = fopen("test_vector.txt", "w");
  //gsl_vector_view testview = gsl_matrix_row(&RBview.matrix, RBview.matrix.size1-1);
  //gsl_vector_fprintf(fp, &testview.vector, "%le");
  //fclose(fp);

  /* NOTE: the above appears to be correct and gives identical values to those in the ipython notebook */

  interp = create_interpolant(&RBview.matrix); 
  
  /* do some timing tests */

  /* create the model dot model weights */
  gsl_matrix *mmw = create_model_model_weights(interp->Bs);

  double randf0 = -0.00004; /* a random frequency to create a model */

  double *modelfull = calloc(wl, sizeof(double));
  double *modelreduced = calloc(nbases, sizeof(double));
  double *timesred = calloc(nbases, sizeof(double));

  /* create model */
  for ( i=0; i<wl; i++ ){ modelfull[i] = sin(2.*M_PI*randf0*gsl_vector_get(times, i)); }
  for ( i=0; i<nbases; i++ ){ modelreduced[i] = sin(2.*M_PI*randf0*gsl_vector_get(times, interp->nodes[i])); }

  gsl_vector_view mfview = gsl_vector_view_array(modelfull, wl);

  struct timespec t1, t2, t3, t4;
  double dt1, dt2;

  /* get the model model term with the full model */
  double mmfull, mmred;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  gsl_blas_ddot(&mfview.vector, &mfview.vector, &mmfull);
  clock_gettime(CLOCK_MONOTONIC, &t2);
  
  clock_gettime(CLOCK_MONOTONIC, &t3);
  mmred = roq_model_dot_model(mmw, modelreduced);
  clock_gettime(CLOCK_MONOTONIC, &t4);

  dt1 = (double)((t2.tv_sec + t2.tv_nsec*1.e-9) - (t1.tv_sec + t1.tv_nsec*1.e-9));
  dt2 = (double)((t4.tv_sec + t4.tv_nsec*1.e-9) - (t3.tv_sec + t3.tv_nsec*1.e-9));
  fprintf(stderr, "M dot M (full) = %le [%.9lf s], M dot M (reduced) = %le [%.9lf s], time ratio = %lf\n", mmfull, dt1, mmred, dt2, dt1/dt2);

  /* get the data dot model terms by generating some random data */
  const gsl_rng_type *T;
  gsl_rng * r;
  
  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc(T);

  double *data = calloc(wl, sizeof(double));
  for ( i=0; i<wl; i++ ){ data[i] = gsl_ran_gaussian(r, 1.0); } 

  /* create the data dot model weights */
  gsl_vector *dmw = create_data_model_weights(interp->Bs, data);
  gsl_vector_view dataview = gsl_vector_view_array(data, wl);
  
  double dmfull, dmred;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  gsl_blas_ddot(&dataview.vector, &mfview.vector, &dmfull);
  clock_gettime(CLOCK_MONOTONIC, &t2);
  
  clock_gettime(CLOCK_MONOTONIC, &t3);
  dmred = roq_data_dot_model(dmw, modelreduced);
  clock_gettime(CLOCK_MONOTONIC, &t4);
  
  dt1 = (double)((t2.tv_sec + t2.tv_nsec*1.e-9) - (t1.tv_sec + t1.tv_nsec*1.e-9));
  dt2 = (double)((t4.tv_sec + t4.tv_nsec*1.e-9) - (t3.tv_sec + t3.tv_nsec*1.e-9));
  fprintf(stderr, "D dot M (full) = %le [%.9lf s], D dot M (reduced) = %le [%.9lf s], time ratio = %lf\n", dmfull, dt1, dmred, dt2, dt1/dt2);

  /* check difference in log likelihoods */
  double Lfull, Lred;
  
  Lfull = mmfull - 2.*dmfull;
  Lred = mmred - 2.*dmred;
  
  fprintf(stderr, "Fractional difference in log likelihoods = %lf%%\n", 100.*fabs(Lfull-Lred)/fabs(Lfull));
  
  return 0;
}
