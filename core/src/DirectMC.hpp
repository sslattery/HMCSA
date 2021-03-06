//---------------------------------------------------------------------------//
// \file DirectMC.hpp
// \author Stuart Slattery
// \brief Direct Monte Carlo solver declaration.
//---------------------------------------------------------------------------//

#ifndef HMCSA_DIRECTMC_HPP
#define HMCSA_DIRECTMC_HPP

#include <Teuchos_RCP.hpp>

#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

namespace HMCSA
{

class DirectMC
{
  private:

    // Linear problem.
    Teuchos::RCP<Epetra_LinearProblem> d_linear_problem;

    // Iteration matrix.
    Epetra_CrsMatrix d_H;

    // Probability matrix.
    Epetra_CrsMatrix d_P;

    // Cumulative distribution function.
    Epetra_CrsMatrix d_C;

  public:

    // Constructor.
    DirectMC( Teuchos::RCP<Epetra_LinearProblem> &linear_problem );

    // Destructor.
    ~DirectMC();

    // Solve.
    void walk( const int num_histories, const double weight_cutoff );

    // Return the iteration matrix.
    const Epetra_CrsMatrix& getH() const
    { return d_H; }

  private:

    // Build the iteration matrix.
    Epetra_CrsMatrix buildH();

    // Build the probability matrix.
    Epetra_CrsMatrix buildP();

    // Build the cumulative distribution function.
    Epetra_CrsMatrix buildC();

};

} // end namespace HMCSA

#endif // end HMCSA_DIRECTMC_HPP

//---------------------------------------------------------------------------//
// end DirectMC.hpp
//---------------------------------------------------------------------------//

