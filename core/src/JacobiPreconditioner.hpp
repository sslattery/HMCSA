//---------------------------------------------------------------------------//
// \file JacobiPreconditioner.hpp
// \author Stuart R. Slattery
// \brief Jacobi preconditioner declaration.
//---------------------------------------------------------------------------//

#ifndef HMCSA_JACOBIPRECONDITIONER_HPP
#define HMCSA_JACOBIPRECONDITIONER_HPP

#include <Teuchos_RCP.hpp>

#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

namespace HMCSA
{

class JacobiPreconditioner
{
  private:

    // Preconditioned operator.
    Teuchos::RCP<Epetra_CrsMatrix> d_M_inv_A;

    // Preconditioned RHS.
    Teuchos::RCP<Epetra_Vector> d_M_inv_b;

  public:

    // Constructor.
    JacobiPreconditioner();

    // Destructor.
    ~JacobiPreconditioner();

    // Do preconditioning.
    void precondition( Teuchos::RCP<Epetra_LinearProblem> &linear_problem );

    // Get the preconditioned Operator.
    const Teuchos::RCP<Epetra_CrsMatrix>& getOperator() const
    { return d_M_inv_A; }

    // Get the preconditioned RHS.
    const Teuchos::RCP<Epetra_Vector>& getRHS() const
    { return d_M_inv_b; }
};

} // End namespace HMCSA

#endif // end HMCSA_JACOBIPRECONDITIONER_HPP

//---------------------------------------------------------------------------//
// end JacobiPreconditioner.hpp
//---------------------------------------------------------------------------//


