//---------------------------------------------------------------------------//
// \file SequentialMC.hpp
// \author Stuart R. Slattery
// \brief Monte Carlo Synthetic Acceleration solver definition.
//---------------------------------------------------------------------------//

#ifndef HSequentialMC_SequentialMC_HPP
#define HMCSA_SequentialMC_HPP

#include <Teuchos_RCP.hpp>

#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

namespace HMCSA
{

class SequentialMC
{
  private:

    // Linear problem.
    Teuchos::RCP<Epetra_LinearProblem> d_linear_problem;

    // Iteration count.
    int d_num_iters;

  public:

    // Constructor.
    SequentialMC( Teuchos::RCP<Epetra_LinearProblem> &linear_problem );

    // Destructor.
    ~SequentialMC();

    // Solve.
    void iterate( const int max_iters,
		  const double tolerance,
		  const int num_histories,
		  const double weight_cutoff );

    // Get the iteration count from the last solve.
    int getNumIters() const
    { return d_num_iters; }
};

} // end namespace HMCSA

#endif // end HMCSA_SequentialMC_HPP

//---------------------------------------------------------------------------//
// end SequentialMC.hpp
//---------------------------------------------------------------------------//

