//---------------------------------------------------------------------------//
// \file JacobiPreconditioner.hpp
// \author Stuart R. Slattery
// \brief Jacobi preconditioner declaration.
//---------------------------------------------------------------------------//

#ifndef HMCSA_JACOBIPRECONDITIONER_HPP
#define HMCSA_JACOBIPRECONDITIONER_HPP

#include <Teuchos_RCP.hpp>

#include <Epetra_LinearProblem.h>

namespace HMCSA
{

class JacobiPreconditioner
{
  public:

    // Constructor.
    JacobiPreconditioner();

    // Destructor.
    ~JacobiPreconditioner();

    // Do preconditioning.
    void precondition( Teuchos::RCP<Epetra_LinearProblem> &linear_problem );
};

} // End namespace HMCSA

#endif // end HMCSA_JACOBIPRECONDITIONER_HPP

//---------------------------------------------------------------------------//
// end JacobiPreconditioner.hpp
//---------------------------------------------------------------------------//


