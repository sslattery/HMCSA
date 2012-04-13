//----------------------------------*-C++-*----------------------------------//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/*!
 * \file   mesh/test/tstEpetra.cpp
 * \author Stuart Slattery
 * \brief  Epetra class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>

#include <Teuchos_UnitTestHarness.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEUCHOS_UNIT_TEST( AztecOO, gmres_test)
{
    int error = 0;

    std::vector<double> x_vector( 3, 0.0 );
    Epetra_SerialComm comm;
    Epetra_Map map( 3, 0, comm );
    Epetra_Vector x( View, map, &x_vector[0] );

    std::vector<double> b_vector( 3 );
    b_vector[0] = 10.0;
    b_vector[1] = 1.0;
    b_vector[2] = 1.0;
    Epetra_Vector b( View, map, &b_vector[0] );

    std::vector<double> A_matrix( 9 );
    A_matrix[0] = 1.0;
    A_matrix[1] = -0.3;
    A_matrix[2] = -0.4;
    A_matrix[3] = -0.2;
    A_matrix[4] = 1.0;
    A_matrix[5] = -0.4;
    A_matrix[6] = -0.5;
    A_matrix[7] = -0.4;
    A_matrix[8] = 1.0;

    std::vector<int> A_indices( 9 );
    for ( int i = 0; i < 2; ++i )
    {
	A_indices[i] = i;
	A_indices[i+3] = i;
	A_indices[i+6] = i;
    }

    std::vector<int> entries_per_row( 3, 3 );
    Epetra_CrsMatrix A( View, map, &entries_per_row[0] );
    A.InsertGlobalValues( 0, 1, &A_matrix[0], &A_indices[0] );
    A.InsertGlobalValues( 0, 1, &A_matrix[1], &A_indices[1] );
    A.InsertGlobalValues( 0, 1, &A_matrix[2], &A_indices[2] );
    A.InsertGlobalValues( 1, 1, &A_matrix[3], &A_indices[3] );
    A.InsertGlobalValues( 1, 1, &A_matrix[4], &A_indices[4] );
    A.InsertGlobalValues( 1, 1, &A_matrix[5], &A_indices[5] );
    A.InsertGlobalValues( 2, 1, &A_matrix[6], &A_indices[6] );
    A.InsertGlobalValues( 2, 1, &A_matrix[7], &A_indices[7] );
    A.InsertGlobalValues( 2, 1, &A_matrix[8], &A_indices[8] );
    error = A.FillComplete();
    TEST_ASSERT( error > 0 );

    
}

//---------------------------------------------------------------------------//
//                        end of tstEpetra.cpp
//---------------------------------------------------------------------------//
