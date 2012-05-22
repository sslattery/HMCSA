//---------------------------------------------------------------------------//
// \file JacobiSolver.cpp
// \author Stuart R. Slattery
// \brief Fixed point Jacobi solver definition.
//---------------------------------------------------------------------------//

#include "JacobiSolver.hpp"
#include "AdjointMC.hpp"

#include <Epetra_Map.h>
#include <Epetra_Vector.h>

namespace HMCSA
{
/*! 
 * \brief Constructor.
 */
JacobiSolver::JacobiSolver( Teuchos::RCP<Epetra_LinearProblem> &linear_problem )
    : d_linear_problem( linear_problem )
    , d_num_iters( 0 )
{ /* ... */ }

/*!
 * \brief Destructor.
 */
JacobiSolver::~JacobiSolver()
{ /* ... */ }
 
/*!
 * \brief Solve.
 */
void JacobiSolver::iterate( const int max_iters, const double tolerance )
{
    // Extract the linear problem.
    Epetra_CrsMatrix *A = 
	dynamic_cast<Epetra_CrsMatrix*>( d_linear_problem->GetMatrix() );
    Epetra_Vector *x = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetLHS() );
    const Epetra_Vector *b = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetRHS() );

    // Setup the residual.
    Epetra_Map row_map = A->RowMap();
    Epetra_Vector residual( row_map );

    // Iterate.
    Epetra_CrsMatrix H = buildH( A );
    Epetra_Vector temp_vec( row_map );
    d_num_iters = 0;
    double residual_norm = 1.0;
    double b_norm;
    b->NormInf( &b_norm );
    double conv_crit = b_norm*tolerance;
    while ( residual_norm > conv_crit && d_num_iters < max_iters )
    {
	H.Apply( *x, temp_vec );
	x->Update( 1.0, temp_vec, 1.0, *b, 0.0 );

	A->Apply( *x, temp_vec );
	residual.Update( 1.0, *b, -1.0, temp_vec, 0.0 );

	residual.NormInf( &residual_norm );
	++d_num_iters;
    }
}

/*!
 * \brief Build the Jacobi iteration matrix.
 */
Epetra_CrsMatrix
JacobiSolver::buildH( const Epetra_CrsMatrix *A )
{
    Epetra_CrsMatrix H(Copy, A->RowMap(), A->GlobalMaxNumEntries() );
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


} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end JacobiSolver.cpp
//---------------------------------------------------------------------------//

