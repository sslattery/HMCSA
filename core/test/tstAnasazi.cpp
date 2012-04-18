//----------------------------------*-C++-*----------------------------------//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/*!
 * \file   mesh/test/tstAnasazi.cpp
 * \author Stuart Slattery
 * \brief  Anasazi class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <ostream>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>

#include "AnasaziMultiVecTraits.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziBlockDavidsonSolMgr.hpp"
#include "AnasaziBasicOutputManager.hpp"
#include "AnasaziEpetraAdapter.hpp"

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEUCHOS_UNIT_TEST( Anasazi, arnoldi_test)
{
    int problem_size = 5;

    Epetra_SerialComm comm;
    Epetra_Map map( problem_size, 0, comm );

    // Build A.
    Epetra_CrsMatrix A( Copy, map, problem_size );
    double lower_diag = -1.0;
    double diag = 2.0;
    double upper_diag = -1.0;
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

    // Block Davidson setup.
    typedef Epetra_MultiVector MV;
    typedef Epetra_Operator OP;
    typedef Anasazi::MultiVecTraits<double, Epetra_MultiVector> MVT;

    const int nev = problem_size;
    const int block_size = 5;
    const int num_blocks = 8;
    const int max_restarts = 100;
    const double tol = 1.0e-8;

    Teuchos::ParameterList solver_params;
    solver_params.set( "Which", "LM" );
    solver_params.set( "Block Size", block_size );
    solver_params.set( "Num Blocks", num_blocks );
    solver_params.set( "Maximum Restarts", max_restarts );
    solver_params.set( "Convergence Tolerance", tol );

    std::vector<double> x_vector( problem_size );
    Epetra_Vector x( View, map, &x_vector[0] );

    Teuchos::RCP<Epetra_MultiVector> ivec 
	= Teuchos::rcp( new Epetra_MultiVector(map, block_size) );
    ivec->Random();

    // Create the eigenproblem.
    Teuchos::RCP<Anasazi::BasicEigenproblem<double, MV, OP> > MyProblem =
	Teuchos::rcp( new Anasazi::BasicEigenproblem<double, MV, OP>(A, ivec) );

    // Inform the eigenproblem that the operator A is symmetric
    MyProblem->setHermitian(true);

    // Set the number of eigenvalues requested.
    MyProblem->setNEV( nev );

    // Create the solver manager
    Anasazi::BlockDavidsonSolMgr<double, MV, OP> MySolverMan(MyProblem, 
							     solver_params);

    // Solve the problem
    Anasazi::ReturnType returnCode = MySolverMan.solve();

    // Get the eigenvalues and eigenvectors from the eigenproblem
    Anasazi::Eigensolution<double,MV> sol = MyProblem->getSolution();
    std::vector<Anasazi::Value<double> > evals = sol.Evals;
    Teuchos::RCP<MV> evecs = sol.Evecs;

    for( int i = 0; i < (int) evals.size(); ++i )
    {
	std::cout << evals[i] << std::endl;
    }
}

//---------------------------------------------------------------------------//
//                        end of tstAnasazi.cpp
//---------------------------------------------------------------------------//
