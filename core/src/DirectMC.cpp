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
DirectMC::DirectMC( Epetra_LinearProblem *linear_problem )
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
    // Setup.
    Epetra_Vector *x = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetLHS() );
    const Epetra_Vector *b = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetRHS() );
    int N = x->GlobalLength();

    // Random walk.
    int state;
    int new_state;
    int new_index;
    double weight;
    double zeta;
    bool walk;
    std::vector<double> H_values( N );
    std::vector<int> H_indices( N );
    int H_size;
    std::vector<double> P_values( N );
    std::vector<int> P_indices( N );
    int size_P;
    std::vector<double> C_values( N );
    std::vector<int> C_indices( N );
    int size_C;
    std::vector<int>::iterator P_it;
    std::vector<int>::iterator H_it;
    for ( int i = 0; i < N; ++i )
    {
	for ( int n = 0; n < num_histories; ++n )
	{
	    state = i;
	    weight = 1.0;
	    walk = true;
	    while ( walk )
	    {
		(*x)[i] += weight * (*b)[state];

		d_C.ExtractGlobalRowCopy( state, 
					  N, 
					  size_C, 
					  &C_values[0], 
					  &C_indices[0] );
		zeta = (double) rand() / RAND_MAX;
		new_index = std::distance( 
		    C_values.begin(),
		    std::lower_bound( C_values.begin(), 
				      C_values.begin()+size_C,
				      zeta ) );
		new_state = C_indices[ new_index ];

		d_P.ExtractGlobalRowCopy( state, 
					  N, 
					  size_P, 
					  &P_values[0], 
					  &P_indices[0] );

		d_H.ExtractGlobalRowCopy( state, 
					  N, 
					  H_size, 
					  &H_values[0], 
					  &H_indices[0] );

		P_it = std::find( P_indices.begin(),
				  P_indices.begin()+size_P,
				  new_state );
		
		H_it = std::find( H_indices.begin(),
				  H_indices.begin()+H_size,
				  new_state );

		if ( P_values[std::distance(P_indices.begin(),P_it)] == 0 ||
		     P_it == P_indices.end() )
		{
		    weight = 0;
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

		state = new_state;
	    }
	}

	(*x)[i] /= num_histories;
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
    std::vector<double> A_values( N );
    std::vector<int> A_indices( N );
    int A_size = 0;
    double local_H;
    bool found_diag = false;
    for ( int i = 0; i < N; ++i )
    {
	A->ExtractGlobalRowCopy( i,
				 N, 
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
    std::vector<double> H_values( N );
    std::vector<int> H_indices( N );
    int H_size = 0;
    double local_P = 0.0;
    for ( int i = 0; i < N; ++i )
    {
	d_H.ExtractGlobalRowCopy( i,
				 N, 
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
	    local_P = abs(H_values[j]) / row_sum;
	    P.InsertGlobalValues( i, 1, &local_P, &H_indices[j] );
	}
    }
    P.FillComplete();
    return P;
}

/*!
 * \brief Build the cumulative distribution function.
 */
Epetra_CrsMatrix DirectMC::buildC()
{
    int N = d_P.NumGlobalRows();
    Epetra_CrsMatrix C( Copy, d_P.RowMap(), d_P.GlobalMaxNumEntries() );
    double local_C = 0.0;
    std::vector<double> P_values( N );
    std::vector<int> P_indices( N );
    int size_P = 0;
    for ( int i = 0; i < N; ++i )
    {
	d_P.ExtractGlobalRowCopy( i, 
				  N, 
				  size_P, 
				  &P_values[0], 
				  &P_indices[0] );
	local_C = 0.0;
	for ( int j = 0; j < size_P; ++j )
	{
	    local_C += P_values[j];
	    C.InsertGlobalValues( i, 1, &local_C, &P_indices[j] );
	}
    }
    C.FillComplete();
    return C;
}

} // namespace HMCSA

//---------------------------------------------------------------------------//
// end DirectMC.cpp
//---------------------------------------------------------------------------//

