//---------------------------------------------------------------------------//
// \file OperatorTools.cpp
// \author Stuart R. Slattery
// \brief OperatorTools definition.
//---------------------------------------------------------------------------//

#include "OperatorTools.hpp"

#include <Teuchos_ParameterList.hpp>

#include <Epetra_Multivector.h>

#include <AnasaziTypes.hpp>
#include <AnasaziBlockDavidsonSlMgr.hpp>

namespace HMCSA
{

/*!
 * \brief Compute the Eigenvalues for the matrix.
 */
std::vector<double> OperatorTools::eigenvalues( const Epetra_Operator& operator )
{
    

}

/*!
 * \brief Compute the spectral radius of an operator.
 */
double OperatorTools::spectralRadius( const Epetra_Operator& operator )
{

}

/*!
 * \brief Compute the stiffness ratio of an operator.
 */
double OperatorTools::stiffnessRatio( const Epetra_Operator& operator )
{

}

} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end OperatorTools.cpp
//---------------------------------------------------------------------------//

