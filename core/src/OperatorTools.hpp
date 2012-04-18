//---------------------------------------------------------------------------//
// \file OperatorTools.hpp
// \author Stuart R. Slattery
// \brief OperatorTools declaration.
//---------------------------------------------------------------------------//

#ifndef HMCSA_OPERATORTOOLS_HPP
#define HMCSA_OPERATORTOOLS_HPP

#include <vector>

#include <Epetra_Operator.hpp>

namespace HMCSA
{

namespace OperatorTools
{

// Compute the Eigenvalues of an operator.
std::vector<double> eigenvalues( const Epetra_Operator& operator );

// Compute the spectral radius of an operator.
double spectralRadius( const Epetra_Operator& operator );

// Compute the stiffness ratio of the operator.
double stiffnessRatio( const Epetra_Operator& operator );

} // end namespace OperatorTools

} // end namespace HMCSA

#endif // HMCSA_OPERATORTOOLS_HPP

//---------------------------------------------------------------------------//
// end OperatorTools.hpp
//---------------------------------------------------------------------------//

