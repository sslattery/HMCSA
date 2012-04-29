//---------------------------------------------------------------------------//
// \file TimeIntegrator.hpp
// \author Stuart R. Slattery
// \brief Time integrator definition.
//---------------------------------------------------------------------------//

#include "TimeIntegrator.hpp"

#include <vector>
#include <string>
#include <sstream>

#include <Epetra_Vector.h>

namespace HMCSA
{

/*!
 * \brief Constructor.
 */
TimeIntegrator::TimeIntegrator( Teuchos::RCP<Epetra_LinearProblem> &linear_problem,
				const VtkWriter &vtk_writer )
    : d_linear_problem( linear_problem )
    , d_solver( d_linear_problem )
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
void TimeIntegrator::integrate( const int num_steps,
				const int max_iters,
				const double tolerance,
				const int num_histories,
				const double weight_cutoff )
{
    for ( int n = 0; n < num_steps; ++n )
    {
	// Solve A u^(n+1) = u^n
	d_solver.iterate( max_iters, tolerance, num_histories, weight_cutoff );
	
	// Write this time step to file.
	writeStep( n );

	// u^n <- u^(n+1)
	buildSource();
    }
}

/*!
 * \brief Build the source.
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
    std::vector<double> step_solution(N);
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

