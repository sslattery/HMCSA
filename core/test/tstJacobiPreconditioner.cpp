//---------------------------------------------------------------------------//
/*!
 * \file   mesh/test/tstPreconditionedJacobi.cpp
 * \author Stuart Slattery
 * \brief  PreconditionedJacobi class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <ostream>

#include "OperatorTools.hpp"
#include "JacobiPreconditioner.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_LinearProblem.h>

#include "EpetraExt_readEpetraLinearSystem.h"

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

// TEUCHOS_UNIT_TEST( PreconditionedJacobi, jacobi_preconditioner_test )
// {
//     int problem_size = 100;

//     Epetra_SerialComm comm;
//     Epetra_Map map( problem_size, 0, comm );

//     // Build A.
//     Teuchos::RCP<Epetra_CrsMatrix> A = 
// 	Teuchos::rcp( new Epetra_CrsMatrix( Copy, map, problem_size ) );
//     double lower_diag = -0.15;
//     double diag = 1.0;
//     double upper_diag = -0.15;
//     int global_row = 0;
//     int lower_row = 0;
//     int upper_row = 0;
//     for ( int i = 0; i < problem_size; ++i )
//     {
// 	global_row = A->GRID(i);
// 	lower_row = i-1;
// 	upper_row = i+1;
// 	if ( lower_row > -1 )
// 	{
// 	    A->InsertGlobalValues( global_row, 1, &lower_diag, &lower_row );
// 	}
// 	A->InsertGlobalValues( global_row, 1, &diag, &global_row );
// 	if ( upper_row < problem_size )
// 	{
// 	    A->InsertGlobalValues( global_row, 1, &upper_diag, &upper_row );
// 	}
//     }
//     A->FillComplete();

//     Teuchos::RCP<Epetra_Vector> bvec = Teuchos::rcp( 
// 	new Epetra_Vector( map, false ) );
//     bvec->Random();

//     Teuchos::RCP<Epetra_Vector> xvec = Teuchos::rcp( 
// 	new Epetra_Vector( map, false ) );
//     xvec->Random();

//     Teuchos::RCP<Epetra_LinearProblem> linear_problem = Teuchos::rcp(
// 	new Epetra_LinearProblem( 
// 	    A.getRawPtr(), xvec.getRawPtr(), bvec.getRawPtr() ) );

//     double spec_rad_A = HMCSA::OperatorTools::spectralRadius( A );
//     std::cout << "Operator Spectral Radius: " 
// 	      << spec_rad_A << std::endl;

//     HMCSA::JacobiPreconditioner preconditioner( linear_problem );
//     preconditioner.precondition();

//     Teuchos::RCP<Epetra_CrsMatrix> precond_A
// 	= preconditioner.getOperator();
//     double spec_rad_precond = 
// 	HMCSA::OperatorTools::spectralRadius( precond_A );
//     std::cout << "Preconditioned Operator Spectral Radius: " 
// 	      << spec_rad_precond << std::endl;
// }

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( OperatorTools, file_test )
{
    Epetra_SerialComm comm;

    Teuchos::RCP<Epetra_CrsMatrix> A;
    EpetraExt::readEpetraLinearSystem( "In_nos1.mtx", comm, &A );

    Epetra_Vector scaling_vec( A->RowMap() ) ;
    A->InvRowSums(scaling_vec);
    A->RightScale(scaling_vec);

    Teuchos::RCP<Epetra_Vector> bvec = Teuchos::rcp( 
	new Epetra_Vector( A->RowMap(), false ) );
    bvec->Random();

    Teuchos::RCP<Epetra_Vector> xvec = Teuchos::rcp( 
	new Epetra_Vector( A->RowMap(), false ) );
    xvec->Random();

    Teuchos::RCP<Epetra_LinearProblem> linear_problem = Teuchos::rcp(
	new Epetra_LinearProblem( 
	    A.getRawPtr(), xvec.getRawPtr(), bvec.getRawPtr() ) );

    HMCSA::JacobiPreconditioner preconditioner( linear_problem );
    preconditioner.precondition();

    Teuchos::RCP<Epetra_CrsMatrix> precond_A
	= preconditioner.getOperator();
    double spec_rad_precond = 
	HMCSA::OperatorTools::spectralRadius( precond_A );
    std::cout << "Preconditioned Operator Spectral Radius: " 
	      << spec_rad_precond << std::endl;
}

//---------------------------------------------------------------------------//
//                        end of tstPreconditionedJacobi.cpp
//---------------------------------------------------------------------------//
