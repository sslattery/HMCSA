//---------------------------------------------------------------------------//
// \file JacobiPreconditioner.cpp
// \author Stuart R. Slattery
// \brief Jacobi preconditioner definition.
//---------------------------------------------------------------------------//

#include "JacobiPreconditioner.hpp"

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>

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
    // Setup.
    const Epetra_CrsMatrix *A = 
	dynamic_cast<Epetra_CrsMatrix*>( d_linear_problem->GetMatrix() );
    Epetra_Vector *x = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetLHS() );
    const Epetra_Vector *b = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetRHS() );
    Epetra_Vector b_cdf = *b;
    int N = x->GlobalLength();
    Epetra_Map map = A->RowMap();

    // Preconditioned system.
    std::vector<int> entries_per_row( N, 1 );
    Epetra_CrsMatrix M_inv_A( Copy, map, &entries_per_row[0] );
    Epetra_Vector M_inv_b( Copy, map );
}

} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end JacobiPreconditioner.cpp
//---------------------------------------------------------------------------//

