//---------------------------------------------------------------------------//
/*!
 * \file   mesh/test/tstOperatorTools.cpp
 * \author Stuart Slattery
 * \brief  OperatorTools class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <ostream>

#include "OperatorTools.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEUCHOS_UNIT_TEST( OperatorTools, spectral_radius_test)
{
    int problem_size = 10;

    Epetra_SerialComm comm;
    Epetra_Map map( problem_size, 0, comm );

    // Build A.
    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, map, problem_size ) );
    double lower_diag = -1.0;
    double diag = 2.0;
    double upper_diag = -1.0;
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
    std::cout << spec_rad_A << std::endl;
}

//---------------------------------------------------------------------------//
//                        end of tstOperatorTools.cpp
//---------------------------------------------------------------------------//
