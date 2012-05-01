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
void buildIC( std::vector<double> &source, 
	      const int xN, const int yN,
	      const double bc_val_xmin, const double bc_val_xmax, 
	      const double bc_val_ymin, const double bc_val_ymax )
{
   int idx;
    for ( int j = 1; j < yN-1; ++j )
    {
	int i = 0;
	idx = i + j*xN;
	source[idx] = bc_val_xmin;
    }
    for ( int j = 1; j < yN-1; ++j )
    {
	int i = xN-1;
	idx = i + j*xN;
	source[idx] = bc_val_xmax;
    }
    for ( int i = 0; i < xN; ++i )
    {
	int j = 0;
	idx = i + j*xN;
	source[idx] = bc_val_ymin;
    }
    for ( int i = 0; i < xN; ++i )
    {
	int j = yN-1;
	idx = i + j*xN;
	source[idx] = bc_val_ymax;
    }
}

//---------------------------------------------------------------------------//
int main( int argc, char** argv )
{
    // Problem parameters.
    int xN = 20;
    int yN = 50;
    int problem_size = xN*yN;
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;
    double ic_val = 10.0;
    double bc_val_xmin = 10.0;
    double bc_val_xmax = 10.0;
    double bc_val_ymin = 0.0;
    double bc_val_ymax = 10.0;
    double dx = 0.001;
    double dy = 0.001;
    double dt = 0.05;
    double alpha = 0.001;
    int num_steps = 2;
    int max_iters = 10000;
    double tolerance = 1.0e-8;
    int num_histories = 100;
    double weight_cutoff = 1.0e-8;

    // Setup up a VTK mesh for output.
    HMCSA::VtkWriter vtk_writer( x_min, x_max, y_min, y_max,
				 dx, dy, xN, yN );

    // Build the Diffusion operator.
    HMCSA::DiffusionOperator diffusion_operator(
	HMCSA::HMCSA_DIRICHLET,
	HMCSA::HMCSA_DIRICHLET,
	HMCSA::HMCSA_DIRICHLET,
	HMCSA::HMCSA_DIRICHLET,
	bc_val_xmin, bc_val_xmax, bc_val_ymin, bc_val_ymax,
	xN, yN,
	dx, dy, dt, alpha );

    Teuchos::RCP<Epetra_CrsMatrix> A = diffusion_operator.getCrsMatrix();
    Epetra_Map map = A->RowMap();

    // Solution Vector.
    std::vector<double> x_vector( problem_size );
    Epetra_Vector x( View, map, &x_vector[0] );
    
    // Build source - set intial and Dirichlet boundary conditions.
    std::vector<double> b_vector( problem_size, ic_val );
    buildIC( b_vector, xN, yN,
	     bc_val_xmin, bc_val_xmax, bc_val_ymin, bc_val_ymax );
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
