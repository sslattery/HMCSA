//----------------------------------*-C++-*----------------------------------//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/*!
 * \file   mesh/test/tstMCSA.cpp
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

TEUCHOS_UNIT_TEST( MCSA, MCSA_test)
{
    int problem_size = 16;

    Epetra_SerialComm comm;
    Epetra_Map map( problem_size, 0, comm );

    std::vector<double> x_vector( problem_size );
    Epetra_Vector x( View, map, &x_vector[0] );

    std::vector<double> x_aztec_vector( problem_size );
    Epetra_Vector x_aztec( View, map, &x_aztec_vector[0] );

    std::vector<double> b_vector( problem_size, 0.4 );
    Epetra_Vector b( View, map, &b_vector[0] );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, map, problem_size ) );
    double lower_diag = -0.1;
    double diag = 2.4;
    double upper_diag = -0.1;
    int global_row = 0;
    int lower_row = 0;
    int upper_row = 0;
    for ( int i = 0; i < problem_size; ++i )
    {
	global_row = A->GRID(i);
	lower_row = i-1;
	upper_row = i+1;
	if ( lower_row > -1 )
	{
	    A->InsertGlobalValues( global_row, 1, &lower_diag, &lower_row );
	}
	A->InsertGlobalValues( global_row, 1, &diag, &global_row );
	if ( upper_row < problem_size )
	{
	    A->InsertGlobalValues( global_row, 1, &upper_diag, &upper_row );
	}
    }
    A->FillComplete();

    double spec_rad_A = HMCSA::OperatorTools::spectralRadius( A );
    std::cout << std::endl <<
	"Operator spectral radius: " << spec_rad_A << std::endl;

    Teuchos::RCP<Epetra_LinearProblem> linear_problem = Teuchos::rcp(
	new Epetra_LinearProblem( A.getRawPtr(), &x, &b ) );

    HMCSA::JacobiPreconditioner preconditioner( linear_problem );
    preconditioner.precondition();

    HMCSA::MCSA mcsa_solver( linear_problem );
    mcsa_solver.iterate( 100, 1.0e-8, 100, 1.0e-8 );
    std::cout << "MCSA ITERS: " << mcsa_solver.getNumIters() << std::endl;

    Epetra_LinearProblem aztec_linear_problem( A.getRawPtr(), &x_aztec, &b );
    AztecOO aztec_solver( aztec_linear_problem );
    aztec_solver.SetAztecOption( AZ_solver, AZ_gmres );
    aztec_solver.Iterate( 100, 1.0e-8 );

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
}

//---------------------------------------------------------------------------//
//                        end of tstMCSA.cpp
//---------------------------------------------------------------------------//
