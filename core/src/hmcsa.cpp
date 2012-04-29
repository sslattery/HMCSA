//---------------------------------------------------------------------------//
// \file hmcsa.cpp
// \author Stuart R. Slattery
// \brief HMCSA executable.
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>

#include "DiffusionOperator.hpp"
#include "TimeIntegrator.hpp"
#include "JacobiPreconditioner.hpp"
#include "HMCSATypes.hpp"
#include "VtkWriter.hpp"

#include <Teuchos_RCP.hpp>

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

//---------------------------------------------------------------------------//
// Build the source with boundary and initial conditions.
void buildIC( std::vector<double> &source, const int N,
	      const double dirichlet_val )
{
   int idx;
    for ( int j = 1; j < N-1; ++j )
    {
	int i = 0;
	idx = i + j*N;
	source[idx] = dirichlet_val;
    }
    for ( int j = 1; j < N-1; ++j )
    {
	int i = N-1;
	idx = i + j*N;
	source[idx] = dirichlet_val;
    }
    for ( int i = 0; i < N; ++i )
    {
	int j = 0;
	idx = i + j*N;
	source[idx] = dirichlet_val;
    }
    for ( int i = 0; i < N; ++i )
    {
	int j = N-1;
	idx = i + j*N;
	source[idx] = dirichlet_val;
    }
}

//---------------------------------------------------------------------------//
int main( int argc, char** argv )
{
    // Problem parameters.
    int N = 10;
    int problem_size = N*N;
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;
    double bc_val = 10.0;
    double dx = 0.01;
    double dy = 0.01;
    double dt = 0.005;
    double alpha = 0.01;
    int num_steps = 2;
    int max_iters = 1000;
    double tolerance = 1.0e-8;
    int num_histories = 100;
    double weight_cutoff = 1.0e-8;

    // Setup up a VTK mesh for output.
    HMCSA::VtkWriter vtk_writer( x_min, x_max, y_min, y_max,
				 dx, dy, N, N );

    // Build the Diffusion operator.
    HMCSA::DiffusionOperator diffusion_operator(
	HMCSA::HMCSA_DIRICHLET,
	HMCSA::HMCSA_DIRICHLET,
	HMCSA::HMCSA_DIRICHLET,
	HMCSA::HMCSA_DIRICHLET,
	bc_val, bc_val, bc_val, bc_val,
	N, N,
	dx, dy, dt, alpha );

    Teuchos::RCP<Epetra_CrsMatrix> A = diffusion_operator.getCrsMatrix();
    Epetra_Map map = A->RowMap();

    // Solution Vector.
    std::vector<double> x_vector( problem_size );
    Epetra_Vector x( View, map, &x_vector[0] );
    
    // Build source - set intial and Dirichlet boundary conditions.
    std::vector<double> b_vector( problem_size, 1.0 );
    buildIC( b_vector, N, bc_val );
    Epetra_Vector b( View, map, &b_vector[0] );

    // Linear problem.
    Teuchos::RCP<Epetra_LinearProblem> linear_problem = Teuchos::rcp(
    	new Epetra_LinearProblem( A.getRawPtr(), &x, &b ) );

    // Jacobi precondition.
    HMCSA::JacobiPreconditioner preconditioner;
    preconditioner.precondition( linear_problem );

    // Time step.
    HMCSA::TimeIntegrator time_integrator( linear_problem, vtk_writer );
    time_integrator.integrate( num_steps, max_iters, 
			       tolerance, num_histories,
			       weight_cutoff );    

    return 0;
}
