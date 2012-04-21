//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mesh/test/tstOneStepSolve.cpp
 * \author Stuart Slattery
 * \brief  Adjoint Monte Carlo solver unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <ostream>

#include "AdjointMC.hpp"
#include "MCSA.hpp"
#include "OperatorTools.hpp"
#include "JacobiPreconditioner.hpp"
#include "DiffusionOperator.hpp"
#include "HMCSATypes.hpp"
#include "VtkWriter.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

#include <AztecOO.h>

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

Teuchos::RCP<Epetra_CrsMatrix>
buildH( const Teuchos::RCP<Epetra_CrsMatrix> &A)
{
    Teuchos::RCP<Epetra_CrsMatrix> H = Teuchos::rcp( 
	new Epetra_CrsMatrix(Copy, A->RowMap(), A->GlobalMaxNumEntries() ) );
    int N = A->NumGlobalRows();
    std::vector<double> A_values( N );
    std::vector<int> A_indices( N );
    int A_size = 0;
    double local_H;
    bool found_diag = false;
    for ( int i = 0; i < N; ++i )
    {
	A->ExtractGlobalRowCopy( i,
				 N, 
				 A_size, 
				 &A_values[0], 
				 &A_indices[0] );

	for ( int j = 0; j < A_size; ++j )
	{
	    if ( i == A_indices[j] )
	    {
		local_H = 1.0 - A_values[j];
		H->InsertGlobalValues( i, 1, &local_H, &A_indices[j] );
		found_diag = true;
	    }
	    else
	    {
		local_H = -A_values[j];
		H->InsertGlobalValues( i, 1, &local_H, &A_indices[j] );
	    }
	}
	if ( !found_diag )
	{
	    local_H = 1.0;
	    H->InsertGlobalValues( i, 1, &local_H, &i );
	}
    }
    H->FillComplete();
    return H;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEUCHOS_UNIT_TEST( MCSA, one_step_solve_test)
{
    int N = 50;
    int problem_size = N*N;

    // Build the diffusion operator.
    double bc_val = 10.0;
    double dx = 0.01;
    double dy = 0.01;
    double dt = 0.05;
    double alpha = 1.0;
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

    // Solution Vectors.
    std::vector<double> x_vector( problem_size );
    Epetra_Vector x( View, map, &x_vector[0] );

    std::vector<double> x_aztec_vector( problem_size );
    Epetra_Vector x_aztec( View, map, &x_aztec_vector[0] );
    
    // Build source - set intial and Dirichlet boundary conditions.
    std::vector<double> b_vector( problem_size, 1.0 );
    int idx;
    for ( int j = 1; j < N-1; ++j )
    {
	int i = 0;
	idx = i + j*N;
	b_vector[idx] = bc_val;
    }
    for ( int j = 1; j < N-1; ++j )
    {
	int i = N-1;
	idx = i + j*N;
	b_vector[idx] = bc_val;
    }
    for ( int i = 0; i < N; ++i )
    {
	int j = 0;
	idx = i + j*N;
	b_vector[idx] = bc_val;
    }
    for ( int i = 0; i < N; ++i )
    {
	int j = N-1;
	idx = i + j*N;
	b_vector[idx] = bc_val;
    }
    Epetra_Vector b( View, map, &b_vector[0] );

    // MCSA Linear problem.
    Teuchos::RCP<Epetra_LinearProblem> linear_problem = Teuchos::rcp(
	new Epetra_LinearProblem( A.getRawPtr(), &x, &b ) );

    // MCSA Jacobi precondition.
    Teuchos::RCP<Epetra_CrsMatrix> H = buildH( A );
    double spec_rad_H = HMCSA::OperatorTools::spectralRadius( H );
    std::cout << std::endl << std::endl
	      << "---------------------" << std::endl
	      << "Iteration matrix spectral radius: " 
	      << spec_rad_H << std::endl;

    HMCSA::JacobiPreconditioner preconditioner;
    preconditioner.precondition( linear_problem );

    H = buildH( preconditioner.getOperator() );
    double spec_rad_precond_H = 
	HMCSA::OperatorTools::spectralRadius( H );
    std::cout << "Preconditioned iteration matrix spectral radius: "
	      << spec_rad_precond_H << std::endl
	      << "---------------------" << std::endl;

    // MCSA Solve.
    HMCSA::MCSA mcsa_solver( linear_problem );
    mcsa_solver.iterate( 10000, 1.0e-12, 80, 1.0e-8 );
    std::cout << "MCSA ITERS: " << mcsa_solver.getNumIters() << std::endl;

    // Aztec GMRES Solve.
    Teuchos::RCP<Epetra_LinearProblem> aztec_linear_problem = Teuchos::rcp(
	new Epetra_LinearProblem( A.getRawPtr(), &x_aztec, &b ) );
    AztecOO aztec_solver( *aztec_linear_problem );
    aztec_solver.SetAztecOption( AZ_solver, AZ_gmres );
    aztec_solver.Iterate( 100, 1.0e-8 );

    // Error comparison.
    std::vector<double> error_vector( problem_size );
    Epetra_Vector error( View, map, &error_vector[0] );
    for (int i = 0; i < problem_size; ++i)
    {
	error[i] = x[i] - x_aztec[i];
    }
    double error_norm;
    error.Norm2( &error_norm );
    std::cout << std::endl << 
	"Aztec GMRES vs. MCSA absolute error L2 norm: " << 
	error_norm << std::endl;    

    // Write the results to file.
    HMCSA::VtkWriter vtk_writer( 0.0, 1.0, 0.0, 1.0,
				 dx, dy, N, N );
    vtk_writer.write_vector( x_vector, "mcsa" );
}

//---------------------------------------------------------------------------//
//                        end of tstOneStepSolve.cpp
//---------------------------------------------------------------------------//
