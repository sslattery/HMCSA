//---------------------------------------------------------------------------//
// \file MCSA.hpp
// \author Stuart R. Slattery
// \brief Monte Carlo Synthetic Acceleration solver definition.
//---------------------------------------------------------------------------//

#ifndef HMCSA_MCSA_HPP
#define HMCSA_MCSA_HPP

#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

namespace HMCSA
{

class MCSA
{
  private:

    // Linear problem.
    Epetra_LinearProblem *d_linear_problem;

    // Iteration count.
    int d_num_iters;

  public:

    // Constructor.
    MCSA( Epetra_LinearProblem *linear_problem );

    // Destructor.
    ~MCSA();

    // Solve.
    void iterate( const int max_iters,
		  const double tolerance,
		  const int num_histories,
		  const double weight_cutoff );

    // Get the iteration count from the last solve.
    int getNumIters() const
    { return d_num_iters; }

  private:

    // Build the iteration matrix.
    Epetra_CrsMatrix buildH();
};

} // end namespace HMCSA

#endif // end HMCSA_MCSA_HPP

//---------------------------------------------------------------------------//
// end MCSA.hpp
//---------------------------------------------------------------------------//

