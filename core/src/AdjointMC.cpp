//---------------------------------------------------------------------------//
// \file AdjointMC.cpp
// \author Stuart Slattery
// \brief Adjoint Monte Carlo solver definition.
//---------------------------------------------------------------------------//

#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>
#include <iterator>
#include <algorithm>

#include "AdjointMC.hpp"

#include <Epetra_Vector.h>
#include <Epetra_Map.h>

namespace HMCSA
{
/*!
 * \brief Constructor.
 */
AdjointMC::AdjointMC( Teuchos::RCP<Epetra_LinearProblem> &linear_problem )
    : d_linear_problem( linear_problem )
    , d_H( buildH() )
    , d_Q( buildQ() )
    , d_C( buildC() )
{ /* ... */ }

/*!
 * \brief Destructor.
 */
AdjointMC::~AdjointMC()
{ /* ... */ }

/*! 
 * \brief Solve.
 */
void AdjointMC::walk( const int num_histories, const double weight_cutoff )
{
    // Get the LHS and source.
    Epetra_Vector *x = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetLHS() );
    const Epetra_Vector *b = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetRHS() );
    int N = x->GlobalLength();
    int n_H = d_H.GlobalMaxNumEntries();
    int n_Q = d_Q.GlobalMaxNumEntries();
    int n_C = d_C.GlobalMaxNumEntries();

    // Setup.
    int state;
    int new_state;
    int init_state;
    int new_index;
    double weight;
    double weight_sign;
    double zeta;
    double relative_cutoff;
    bool walk;
    std::vector<double> b_cdf( N );
    std::vector<double> H_values( n_H );
    std::vector<int> H_indices( n_H );
    int H_size;
    std::vector<double> Q_values( n_Q );
    std::vector<int> Q_indices( n_Q );
    int Q_size;
    std::vector<double> C_values( n_C );
    std::vector<int> C_indices( n_C );
    int C_size;
    std::vector<int>::iterator Q_it;
    std::vector<int>::iterator H_it;

    // Build source cdf.
    b_cdf[0] = std::abs((*b)[0]);
    double b_norm = b_cdf[0];
    for ( int i = 1; i < N; ++i )
    {
	b_norm += std::abs((*b)[i]);
	b_cdf[i] = std::abs((*b)[i]) + b_cdf[i-1];	
    }
    for ( int i = 0; i < N; ++i )
    {
	b_cdf[i] /= b_norm;
    }

    // Do random walks for specified number of histories.
    bool use_collision_estimator = true;
    for ( int n = 0; n < num_histories; ++n )
    {
	// Sample the source to get the initial state.
	zeta = (double) rand() / RAND_MAX;
	init_state = std::distance( 
	    b_cdf.begin(),
	    std::lower_bound( b_cdf.begin(), b_cdf.end(), zeta ) );

	// Random walk.
	weight = b_norm;
	relative_cutoff = weight_cutoff*weight;
	weight_sign = (*b)[init_state] / std::abs( (*b)[init_state] );
	state = init_state;
	walk = true;
	while ( walk )
	{
            // Get the current iteration matrix row.
	    d_H.ExtractGlobalRowCopy( state, 
				      n_H, 
				      H_size, 
				      &H_values[0], 
				      &H_indices[0] );

	    // Update LHS with either the collision estimator.
            if ( use_collision_estimator )
            {
                (*x)[state] += weight_sign * weight / num_histories;
            }
            // Or the expected value estimator.
            else
            {
                for ( int j = 0; j < H_size; ++j )
                {
                    (*x)[ H_indices[j] ] += 
                        weight_sign * weight * H_values[j] / num_histories;
                }
            }

	    // Sample the CDF to get the next state.
	    d_C.ExtractGlobalRowCopy( state, 
	    			      n_C, 
	    			      C_size, 
	    			      &C_values[0], 
	    			      &C_indices[0] );

	    zeta = (double) rand() / RAND_MAX;
	    new_index = std::distance( 
	    	C_values.begin(),
	    	std::lower_bound( C_values.begin(), 
	    			  C_values.begin()+C_size,
	    			  zeta ) );
	    new_state = C_indices[ new_index ];
	    
	    // Get the state components of Q and H.
	    d_Q.ExtractGlobalRowCopy( state, 
				      n_Q, 
				      Q_size, 
				      &Q_values[0], 
				      &Q_indices[0] );

	    d_H.ExtractGlobalRowCopy( new_state, 
				      n_H, 
				      H_size, 
				      &H_values[0], 
				      &H_indices[0] );

	    Q_it = std::find( Q_indices.begin(),
			      Q_indices.begin()+Q_size,
			      new_state );

	    H_it = std::find( H_indices.begin(),
			      H_indices.begin()+H_size,
			      state );

	    // Compute new weight.
	    if ( Q_values[std::distance(Q_indices.begin(),Q_it)] == 0 ||
		 Q_it == Q_indices.end() ||
		 H_it == H_indices.begin()+H_size )
	    {
		weight = 0.0;
	    }
	    else
	    {
		weight *= H_values[std::distance(H_indices.begin(),H_it)] / 
			  Q_values[std::distance(Q_indices.begin(),Q_it)];
	    }

	    if ( weight < relative_cutoff )
	    {
		walk = false;
	    }

	    // Update the state.
	    state = new_state;
	}
    }

    // If using the expected value estimator, add the source.
    if ( !use_collision_estimator )
    {
        x->Update( 1.0, *b, 1.0 );
    }
}

/*!
 * \brief Build the iteration matrix.
 */
Epetra_CrsMatrix AdjointMC::buildH()
{
    const Epetra_CrsMatrix *A = 
	dynamic_cast<Epetra_CrsMatrix*>( d_linear_problem->GetMatrix() );
    Epetra_CrsMatrix H( Copy, A->RowMap(), A->GlobalMaxNumEntries() );
    int N = A->NumGlobalRows();
    int n_A = A->GlobalMaxNumEntries();
    std::vector<double> A_values( n_A );
    std::vector<int> A_indices( n_A );
    int A_size = 0;
    double local_H;
    bool found_diag = false;
    for ( int i = 0; i < N; ++i )
    {
	A->ExtractGlobalRowCopy( i,
				 n_A, 
				 A_size, 
				 &A_values[0], 
				 &A_indices[0] );

	for ( int j = 0; j < A_size; ++j )
	{
	    if ( i == A_indices[j] )
	    {
		local_H = 1.0 - A_values[j];
		H.InsertGlobalValues( i, 1, &local_H, &A_indices[j] );
		found_diag = true;
	    }
	    else
	    {
		local_H = -A_values[j];
		H.InsertGlobalValues( i, 1, &local_H, &A_indices[j] );
	    }

	    if ( !found_diag )
	    {
		local_H = 1.0;
		H.InsertGlobalValues( i, 1, &local_H, &i );
	    }
	}
    }

    H.FillComplete();
    H.OptimizeStorage();
    return H;
}

/*!
 * \brief Build the adjoint probability matrix.
 */
Epetra_CrsMatrix AdjointMC::buildQ()
{
    Epetra_CrsMatrix Q( Copy, d_H.RowMap(), d_H.GlobalMaxNumEntries() );
    int N = d_H.NumGlobalRows();
    int n_H = d_H.GlobalMaxNumEntries();
    std::vector<double> H_values( n_H );
    std::vector<int> H_indices( n_H );
    int H_size = 0;
    double local_Q = 0.0;
    double row_sum = 0.0;
    for ( int i = 0; i < N; ++i )
    {
	d_H.ExtractGlobalRowCopy( i,
				   n_H, 
				   H_size, 
				   &H_values[0], 
				   &H_indices[0] );

	row_sum = 0.0;
	for ( int j = 0; j < H_size; ++j )
	{
	    row_sum += std::abs(H_values[j]);
	}

	for ( int j = 0; j < H_size; ++j )
	{
	    if ( row_sum > 0.0 )
	    {
		local_Q = std::abs(H_values[j]) / row_sum;
	    }
	    else
	    {
		local_Q = 0.0;
	    }
	    Q.InsertGlobalValues( H_indices[j], 1, &local_Q, &i );
	}
    }

    Q.FillComplete();
    Q.OptimizeStorage();
    return Q;
}

/*!
 * \brief Build the cumulative distribution function.
 */
Epetra_CrsMatrix AdjointMC::buildC()
{
    int N = d_Q.NumGlobalRows();
    int n_Q = d_Q.GlobalMaxNumEntries();
    Epetra_CrsMatrix C( Copy, d_Q.RowMap(), d_Q.GlobalMaxNumEntries() );
    double local_C = 0.0;
    std::vector<double> Q_values( n_Q );
    std::vector<int> Q_indices( n_Q );
    int size_Q = 0;
    for ( int i = 0; i < N; ++i )
    {
	d_Q.ExtractGlobalRowCopy( i, 
				  n_Q, 
				  size_Q, 
				  &Q_values[0], 
				  &Q_indices[0] );

	local_C = 0.0;
	for ( int j = 0; j < size_Q; ++j )
	{
	    local_C += Q_values[j];
	    C.InsertGlobalValues( i, 1, &local_C, &Q_indices[j] );
	}
    }

    C.FillComplete();
    C.OptimizeStorage();
    return C;
}

} // namespace HMCSA

//---------------------------------------------------------------------------//
// end AdjointMC.cpp
//---------------------------------------------------------------------------//

