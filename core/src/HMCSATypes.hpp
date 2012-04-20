//---------------------------------------------------------------------------//
// \file HMCSATypes.hpp
// \author Stuart R. Slattery
// \brief Types for HMCSA.
//---------------------------------------------------------------------------//

#ifndef HMCSA_TYPES_HPP
#define HMCSA_TYPES_HPP

namespace HMCSA
{

// Boundary condition type.
enum HMCSA_BoundaryType {
    HMCSA_BoundaryType_MIN = 0,
    HMCSA_DIRICHLET = HMCSA_BoundaryType_MIN,
    HMCSA_NEUMANN,
    HMCSA_BoundaryType_MAX = HMCSA_NEUMANN
};

} // end namespace HMCSA

#endif // end HMCSA_TYPES_HPP

//---------------------------------------------------------------------------//
// end HMCSATypes.hpp
//---------------------------------------------------------------------------//

