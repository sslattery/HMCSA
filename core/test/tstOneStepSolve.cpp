//----------------------------------*-C++-*----------------------------------//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
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

#include "MCSA.hpp"
#include "OperatorTools.hpp"
#include "JacobiPreconditioner.hpp"
#include "DiffusionOperator.hpp"
#include "HMCSATypes.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

#include <AztecOO.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEUCHOS_UNIT_TEST( MCSA, one_step_solve_test)
{
    int N = 100;
    int problem_size = N*N;

    // Build operator.
    double bc_val = 10.0;
    double dx = 0.01;
    double dy = 0.01;
    double dt = 0.05;
    double alpha = 1.0;
    HMCSA::DiffusionOperator diffusion_operator( HMCSA::HMCSA_DIRICHLET,
						 HMCSA::HMCSA_DIRICHLET,
						 HMCSA::HMCSA_DIRICHLET,
						 HMCSA::HMCSA_DIRICHLET,
						 bc_val, bc_val, bc_val, bc_val,
						 N, N,
						 dx, dy, dt, alpha );

    Teuchos::RCP<Epetra_CrsMatrix> A = diffusion_operator.getCrsMatrix();

    double spec_rad_A = HMCSA::OperatorTools::spectralRadius( A );
    std::cout << std::endl <<
	"Operator spectral radius: " << spec_rad_A << std::endl;

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

    // MCSA Precondition.
    HMCSA::JacobiPreconditioner preconditioner;
    preconditioner.precondition( linear_problem );

    double spec_rad_precond_A = 
	HMCSA::OperatorTools::spectralRadius( preconditioner.getOperator() );
    std::cout << std::endl << "Preconditioned Operator spectral radius: " 
	      << spec_rad_precond_A << std::endl;

    // MCSA Solve.
    HMCSA::MCSA mcsa_solver( linear_problem );
    mcsa_solver.iterate( 100, 1.0e-8, 100, 1.0e-8 );
    std::cout << "MCSA ITERS: " << mcsa_solver.getNumIters() << std::endl;

    // Aztec GMRES Solve.
    Epetra_LinearProblem aztec_linear_problem( A.getRawPtr(), &x_aztec, &b );
    AztecOO aztec_solver( aztec_linear_problem );
    aztec_solver.SetAztecOption( AZ_solver, AZ_gmres );
    aztec_solver.Iterate( 100, 1.0e-8 );

    // Error comparison.
    std::vector<double> error_vector( problem_size );
    Epetra_Vector error( View, map, &error_vector[0] );
    for (int i = 0; i < problem_size; ++i)
    {
	std::cout << x_aztec[i] << std::endl;
	error[i] = x[i] - x_aztec[i];
    }
    double error_norm;
    error.Norm2( &error_norm );
    std::cout << std::endl << 
	"Aztec GMRES vs. MCSA absolute error L2 norm: " << 
	error_norm << std::endl;    
}

//---------------------------------------------------------------------------//
//                        end of tstOneStepSolve.cpp
//---------------------------------------------------------------------------//
