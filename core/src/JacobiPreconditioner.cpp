//---------------------------------------------------------------------------//
// \file JacobiPreconditioner.cpp
// \author Stuart R. Slattery
// \brief Jacobi preconditioner definition.
//---------------------------------------------------------------------------//

#include <vector>

#include "JacobiPreconditioner.hpp"

#include <Epetra_Map.h>

namespace HMCSA
{

/*!
 * \brief Constructor.
 */
JacobiPreconditioner::JacobiPreconditioner()
{ /* ... */ }

/*!
 * \brief Destructor.
 */
JacobiPreconditioner::~JacobiPreconditioner()
{ /* ... */ }

/*!
 * \brief Do preconditioning.
 */
void JacobiPreconditioner::precondition( 
    Teuchos::RCP<Epetra_LinearProblem> &linear_problem )
{
    // Get unconditioned system.
    const Epetra_CrsMatrix *A = 
	dynamic_cast<Epetra_CrsMatrix*>( linear_problem->GetMatrix() );
    const Epetra_Vector *b = 
	dynamic_cast<Epetra_Vector*>( linear_problem->GetRHS() );
    int N = b->GlobalLength();
    Epetra_Map map = A->RowMap();

    // Setup preconditioned system.
    std::vector<int> entries_per_row( N, 1 );
    d_M_inv_A = Teuchos::rcp( 
	new Epetra_CrsMatrix( Copy, map, &entries_per_row[0] ) );
    d_M_inv_b = Teuchos::rcp(
	new Epetra_Vector( map, false ) );

    // Get the diagonal from A.
    Epetra_Vector A_diag( map, false );
    A->ExtractDiagonalCopy( A_diag );

    // Compute (M^-1 A) and (M^-1 b)
    double ma_val;
    int A_size;
    std::vector<double> A_values(N);
    std::vector<int> A_indices(N);
    for ( int i = 0; i < N; ++i )
    {
	A->ExtractGlobalRowCopy( i, N, A_size, &A_values[0], &A_indices[0] );
	for ( int j = 0; j < A_size; ++j )
	{
	    ma_val = A_values[j] / A_diag[i];
	    d_M_inv_A->InsertGlobalValues( i, 1, &ma_val, &A_indices[j] );
	}

	(*d_M_inv_b)[i] = (*b)[i] / A_diag[i];
    }

    d_M_inv_A->FillComplete();

    // Modify the linear problem with the preconditioned system.
    linear_problem->SetOperator( d_M_inv_A.getRawPtr() );
    linear_problem->SetRHS( d_M_inv_b.getRawPtr() );
}

} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end JacobiPreconditioner.cpp
//---------------------------------------------------------------------------//

