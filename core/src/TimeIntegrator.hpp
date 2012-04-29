//---------------------------------------------------------------------------//
// \file TimeIntegrator.hpp
// \author Stuart R. Slattery
// \brief Time integrator declaration.
//---------------------------------------------------------------------------//

#ifndef HMCSA_TIMEINTEGRATOR_HPP
#define HMCSA_TIMEINTEGRATOR_HPP

#include "MCSA.hpp"
#include "VtkWriter.hpp"

#include <Teuchos_RCP.hpp>

#include <Epetra_LinearProblem.h>

namespace HMCSA
{

class TimeIntegrator
{
  private:

    // Linear problem.
    Teuchos::RCP<Epetra_LinearProblem> d_linear_problem;

    // MCSA solver.
    MCSA d_solver;

    // VTK writer.
    VtkWriter d_vtk_writer;

  public:

    // Constructor.
    TimeIntegrator( Teuchos::RCP<Epetra_LinearProblem> &linear_problem,
		    const VtkWriter &vtk_writer );

    // Destructor.
    ~TimeIntegrator();

    // Time step.
    void integrate( const int num_steps,
		    const int max_iters,
		    const double tolerance,
		    const int num_histories,
		    const double weight_cutoff );

  private:

    // Build the source.
    void buildSource();

    // Write the time step to file.
    void writeStep( const int step_number );
};

} // end namespace HMCSA

#endif // end HMCSA_TIMEINTEGRATOR_HPP

//---------------------------------------------------------------------------//
// end TimeIntegrator.hpp
//---------------------------------------------------------------------------//

