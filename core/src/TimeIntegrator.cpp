//---------------------------------------------------------------------------//
// \file TimeIntegrator.hpp
// \author Stuart R. Slattery
// \brief Time integrator definition.
//---------------------------------------------------------------------------//

#include "TimeIntegrator.hpp"

#include <ctime>
#include <cstdio>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <Epetra_Vector.h>

namespace HMCSA
{

/*!
 * \brief Constructor.
 */
TimeIntegrator::TimeIntegrator( 
    Teuchos::RCP<Epetra_LinearProblem> &linear_problem,
    const VtkWriter &vtk_writer )
    : d_linear_problem( linear_problem )
    , d_solver( d_linear_problem )
    , d_preconditioner( d_linear_problem )
    , d_vtk_writer( vtk_writer )
{ /* ... */ }

/*! 
 * \brief Destructor
 */
TimeIntegrator::~TimeIntegrator()
{ /* ... */ }

/*!
 * \brief Time step.
 */
void TimeIntegrator::integrate( bool use_adjoint,
				const int num_steps,
				const int max_iters,
				const double tolerance,
				const int num_histories,
				const double weight_cutoff )
{
    // Precondition the operator before we start time stepping.
    d_preconditioner.preconditionOperator();

    // Setup a time value for timing.
    std::clock_t start, end;
    double timer;
    
    // Do time steps.
    for ( int n = 0; n < num_steps; ++n )
    {
	// Jacobi precondition the RHS. The operator doesn't change.
	d_preconditioner.preconditionRHS();

	// Solve A u^(n+1) = u^n
	start = clock();
	d_solver.iterate( max_iters, tolerance, 
			  num_histories, weight_cutoff );
	end = clock();
	timer = (double)(end - start) / CLOCKS_PER_SEC;

	// Write this time step to file.
	writeStep( n );

	// u^n <- u^(n+1)
	buildSource();

	// Output.
	std::cout << "TIME STEP " << n << ": " << d_solver.getNumIters()
		  << " MCSA iterations    " 
		  << timer << " seconds" << std::endl;
    }
}

/*!
 * \brief Build the source. u^n <- u^(n+1)
 */
void TimeIntegrator::buildSource()
{
    Epetra_Vector const *x = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetLHS() );
    Epetra_Vector *b = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetRHS() );

    b->Update( 1.0, *x, 0.0 );
}

/*!
 * \brief Write the time step to file.
 */
void TimeIntegrator::writeStep( const int step_number )
{
    Epetra_Vector const *x = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetLHS() );

    int N = x->GlobalLength();
    std::vector<double> step_solution( N );
    x->ExtractCopy( &step_solution[0] );

    std::stringstream convert;
    std::string name;
    convert << step_number;
    name = convert.str();
    d_vtk_writer.write_vector( step_solution, name );
}

} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end TimeIntegrator.hpp
//---------------------------------------------------------------------------//

