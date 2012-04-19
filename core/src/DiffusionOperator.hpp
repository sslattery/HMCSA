//---------------------------------------------------------------------------//
// \file DiffusionOperator.hpp
// \author Stuart R. Slattery
// \brief Diffusion operator declaration.
//---------------------------------------------------------------------------//

#ifndef HMCSA_DIFFUSIONOPERATOR_HPP
#define HMCSA_DIFFUSIONOPERATOR_HPP

#include <cstdlib>

#include <Epetra_CrsMatrix.h>

namespace HMCSA
{

class DiffusionOperator
{
  private:

    // X min boundary type.
    std::size_t d_x_min_type;

    // X max boundary type.
    std::size_t d_x_max_type;

    // Y min boundary type.
    std::size_t d_y_min_type;

    // Y max boundary type.
    std::size_t d_y_max_type;

    // X min boundary value.
    double d_x_min_value;

    // X max boundary value.
    double d_x_max_value;

    // Y min boundary value.
    double d_y_min_value;

    // Y max boundary value.
    double d_y_max_value;

    // Operator.
    Epetra_CrsMatrix d_matrix;

  public:

    // Constructor.
    DiffusionOperator( const int x_min_type,
		       const int x_max_type,
		       const int y_min_type,
		       const int y_max_type,
		       const double x_min_value,
		       const double x_max_value,
		       const double y_min_value,
		       const double y_max_value,
		       const int size_x,
		       const int size_y );

    // Destructor.
    ~DiffusionOperator();

    // Get the operator.
    const CrsMatrix& getCrsMatrix() const
    { return d_matrix };

  private:

    // Build the diffusion operator.
    CrsMatrix build_diffusion_operator( const int size_x, const int size_y );
};

} // end namespace HMCSA

#endif // end HMCSA_DIFFUSIONOPERATOR_HPP

//---------------------------------------------------------------------------//
// end DiffusionOperator.hpp
//---------------------------------------------------------------------------//

