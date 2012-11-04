//---------------------------------------------------------------------------//
// \file DirectMC.cpp
// \author Stuart Slattery
// \brief Direct Monte Carlo solver definition.
//---------------------------------------------------------------------------//

#include <cstdlib>
#include <cmath>
#include <vector>
#include <iterator>
#include <algorithm>

#include "DirectMC.hpp"

#include <Epetra_Vector.h>

namespace HMCSA
{
/*!
 * \brief Constructor.
 */
DirectMC::DirectMC( Teuchos::RCP<Epetra_LinearProblem> &linear_problem )
    : d_linear_problem( linear_problem )
    , d_H( buildH() )
    , d_P( buildP() )
    , d_C( buildC() )
{ /* ... */ }

/*!
 * \brief Destructor.
 */
DirectMC::~DirectMC()
{ /* ... */ }

/*! 
 * \brief Solve.
 */
void DirectMC::walk( const int num_histories, const double weight_cutoff )
{
    // Get the LHS and source.
    Epetra_Vector *x = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetLHS() );
    const Epetra_Vector *b = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetRHS() );
    int N = x->GlobalLength();
    int n_H = d_H.GlobalMaxNumEntries();
    int n_P = d_P.GlobalMaxNumEntries();
    int n_C = d_C.GlobalMaxNumEntries();

    // Setup.
    int state;
    int new_state;
    int new_index;
    double weight;
    double zeta;
    bool walk;
    std::vector<double> H_values( n_H );
    std::vector<int> H_indices( n_H );
    int H_size;
    std::vector<double> P_values( n_P );
    std::vector<int> P_indices( n_P );
    int P_size;
    std::vector<double> C_values( n_C );
    std::vector<int> C_indices( n_C );
    int C_size;
    std::vector<int>::iterator P_it;
    std::vector<int>::iterator H_it;

    // Do random walks for specified number of histories.
    for ( int i = 0; i < N; ++i )
    {
	for ( int n = 0; n < num_histories; ++n )
	{
	    // Random walk.
	    state = i;
	    weight = 1.0;
	    walk = true;
	    while ( walk )
	    {
		// Update LHS.
		(*x)[i] += weight * (*b)[state] / num_histories;

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

		// Get the state components of P and H.
		d_P.ExtractGlobalRowCopy( state, 
					  n_P, 
					  P_size, 
					  &P_values[0], 
					  &P_indices[0] );

		d_H.ExtractGlobalRowCopy( state, 
					  n_H, 
					  H_size, 
					  &H_values[0], 
					  &H_indices[0] );

		P_it = std::find( P_indices.begin(),
				  P_indices.begin()+P_size,
				  new_state );
		
		H_it = std::find( H_indices.begin(),
				  H_indices.begin()+H_size,
				  new_state );

		// Compute the new weight.
		if ( P_values[std::distance(P_indices.begin(),P_it)] == 0 ||
		     P_it == P_indices.end() ||
		     H_it == H_indices.begin()+H_size )
		{
		    weight = 0.0;
		}
		else
		{
		    weight *= H_values[std::distance(H_indices.begin(),H_it)] / 
			      P_values[std::distance(P_indices.begin(),P_it)];
		}

		if ( weight < weight_cutoff )
		{
		    walk = false;
		}

		// Update the state.
		state = new_state;
	    }
	}
    }
}

/*!
 * \brief Build the iteration matrix.
 */
Epetra_CrsMatrix DirectMC::buildH()
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
	}
	if ( !found_diag )
	{
	    local_H = 1.0;
	    H.InsertGlobalValues( i, 1, &local_H, &i );
	}
    }

    H.FillComplete();
    H.OptimizeStorage();
    return H;
}

/*!
 * \brief Build the probability matrix.
 */
Epetra_CrsMatrix DirectMC::buildP()
{
    Epetra_CrsMatrix P( Copy, d_H.RowMap(), d_H.GlobalMaxNumEntries() );
    double row_sum = 0.0;
    int N = d_H.NumGlobalRows();
    int n_H = d_H.GlobalMaxNumEntries();
    std::vector<double> H_values( n_H );
    std::vector<int> H_indices( n_H );
    int H_size = 0;
    double local_P = 0.0;
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
	    row_sum += abs(H_values[j]);
	}
	for ( int j = 0; j < H_size; ++j )
	{
	    if ( row_sum == 0.0 )
	    {
		local_P = row_sum;
	    }
	    else
	    {
		local_P = abs(H_values[j]) / row_sum;
	    }
	    P.InsertGlobalValues( i, 1, &local_P, &H_indices[j] );
	}
    }

    P.FillComplete();
    P.OptimizeStorage();
    return P;
}

/*!
 * \brief Build the cumulative distribution function.
 */
Epetra_CrsMatrix DirectMC::buildC()
{
    int N = d_P.NumGlobalRows();
    int n_P = d_P.GlobalMaxNumEntries();
    Epetra_CrsMatrix C( Copy, d_P.RowMap(), d_P.GlobalMaxNumEntries() );
    double local_C = 0.0;
    std::vector<double> P_values( n_P );
    std::vector<int> P_indices( n_P );
    int P_size = 0;
    for ( int i = 0; i < N; ++i )
    {
	d_P.ExtractGlobalRowCopy( i, 
				  n_P, 
				  P_size, 
				  &P_values[0], 
				  &P_indices[0] );
	local_C = 0.0;
	for ( int j = 0; j < P_size; ++j )
	{
	    local_C += P_values[j];
	    C.InsertGlobalValues( i, 1, &local_C, &P_indices[j] );
	}
    }

    C.FillComplete();
    C.OptimizeStorage();
    return C;
}

} // namespace HMCSA

//---------------------------------------------------------------------------//
// end DirectMC.cpp
//---------------------------------------------------------------------------//

