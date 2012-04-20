//---------------------------------------------------------------------------//
/*!
 * \file   mesh/test/tstDiffusionOperator.cpp
 * \author Stuart Slattery
 * \brief  DiffusionOperator class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <ostream>

#include "HMCSATypes.hpp"
#include "DiffusionOperator.hpp"
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

TEUCHOS_UNIT_TEST( DiffusionOperator, diffusion_operator_radius_test )
{
    HMCSA::DiffusionOperator diffusion_operator( HMCSA::HMCSA_DIRICHLET,
						 HMCSA::HMCSA_DIRICHLET,
						 HMCSA::HMCSA_DIRICHLET,
						 HMCSA::HMCSA_DIRICHLET,
						 1.0, 1.0, 1.0, 1.0,
						 4, 4,
						 0.01, 0.01, 0.05, 1.0 );

    Teuchos::RCP<Epetra_CrsMatrix> matrix = diffusion_operator.getCrsMatrix();

    double spec_rad_matrix = HMCSA::OperatorTools::spectralRadius( matrix );
    std::cout << "Diffusion Operator Spectral Radius: " 
	      << spec_rad_matrix << std::endl;
}

//---------------------------------------------------------------------------//
//                        end of tstDiffusionOperator.cpp
//---------------------------------------------------------------------------//
