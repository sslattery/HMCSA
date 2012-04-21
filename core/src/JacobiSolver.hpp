//---------------------------------------------------------------------------//
// \file JacobiSolver.hpp
// \author Stuart R. Slattery
// \brief Fixed point Jacobi solver declaration.
//---------------------------------------------------------------------------//

#ifndef HMCSA_JacobiSolver_HPP
#define HMCSA_JacobiSolver_HPP

#include <Teuchos_RCP.hpp>

#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

namespace HMCSA
{

class JacobiSolver
{
  private:

    // Linear problem.
    Teuchos::RCP<Epetra_LinearProblem> d_linear_problem;

    // Iteration count.
    int d_num_iters;

  public:

    // Constructor.
    JacobiSolver( Teuchos::RCP<Epetra_LinearProblem> &linear_problem );

    // Destructor.
    ~JacobiSolver();

    // Solve.
    void iterate( const int max_iters, const double tolerance );

    // Get the iteration count from the last solve.
    int getNumIters() const
    { return d_num_iters; }

  private:

    // brief Build the Jacobi iteration matrix.
    Epetra_CrsMatrix buildH( const Epetra_CrsMatrix *A );

};

} // end namespace HMCSA

#endif // end HMCSA_JacobiSolver_HPP

//---------------------------------------------------------------------------//
// end JacobiSolver.hpp
//---------------------------------------------------------------------------//

