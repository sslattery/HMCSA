//----------------------------------*-C++-*----------------------------------//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/*!
 * \file   mesh/test/tstAdjointMC.cpp
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

#include <Teuchos_UnitTestHarness.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

#include <AztecOO.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEUCHOS_UNIT_TEST( AdjointMC, AdjointMC_test)
{
    int problem_size = 6;

    Epetra_SerialComm comm;
    Epetra_Map map( problem_size, 0, comm );

    std::vector<double> x_vector( problem_size );
    Epetra_Vector x( View, map, &x_vector[0] );

    std::vector<double> x_aztec_vector( problem_size );
    Epetra_Vector x_aztec( View, map, &x_aztec_vector[0] );

    std::vector<double> b_vector( problem_size, 0.4 );
    Epetra_Vector b( View, map, &b_vector[0] );

    Epetra_CrsMatrix A( Copy, map, problem_size );
    double lower_diag = -0.4;
    double diag = 1.0;
    double upper_diag = -0.4;
    int global_row = 0;
    int lower_row = 0;
    int upper_row = 0;
    for ( int i = 0; i < problem_size; ++i )
    {
	global_row = A.GRID(i);
	lower_row = i-1;
	upper_row = i+1;
	if ( lower_row > -1 )
	{
	    A.InsertGlobalValues( global_row, 1, &lower_diag, &lower_row );
	}
	A.InsertGlobalValues( global_row, 1, &diag, &global_row );
	if ( upper_row < problem_size )
	{
	    A.InsertGlobalValues( global_row, 1, &upper_diag, &upper_row );
	}
    }
    A.FillComplete();

    Epetra_LinearProblem *linear_problem = 
	new Epetra_LinearProblem( &A, &x, &b );

    HMCSA::AdjointMC adjoint_mc_solver( linear_problem );
    adjoint_mc_solver.walk( 1000, 1.0e-8 );

    for (int i = 0; i < problem_size; ++i)
    {
	std::cout << x_vector[i] << std::endl;
    }

    Epetra_LinearProblem aztec_linear_problem( &A, &x_aztec, &b );

    AztecOO aztec_solver( aztec_linear_problem );
    aztec_solver.SetAztecOption( AZ_solver, AZ_gmres );
    aztec_solver.Iterate( 100, 1.0e-8 );

    std::cout << std::endl;
    std::cout << "Adjoint MC Solution" << std::endl;
    for (int i = 0; i < problem_size; ++i)
    {
	std::cout << x_vector[i] << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Aztec Solution" << std::endl;
    for (int i = 0; i < problem_size; ++i)
    {
	std::cout << x_aztec_vector[i] << std::endl;
    }
}

//---------------------------------------------------------------------------//
//                        end of tstAdjointMC.cpp
//---------------------------------------------------------------------------//
