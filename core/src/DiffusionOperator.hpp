//---------------------------------------------------------------------------//
// \file DiffusionOperator.hpp
// \author Stuart R. Slattery
// \brief Second order diffusion operator declaration.
//---------------------------------------------------------------------------//

#ifndef HMCSA_DIFFUSIONOPERATOR_HPP
#define HMCSA_DIFFUSIONOPERATOR_HPP

#include <cstdlib>

#include <Teuchos_RCP.hpp>

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
    Teuchos::RCP<Epetra_CrsMatrix> d_matrix;

  public:

    // Constructor.
    DiffusionOperator( const int stencil,
		       const int x_min_type,
		       const int x_max_type,
		       const int y_min_type,
		       const int y_max_type,
		       const double x_min_value,
		       const double x_max_value,
		       const double y_min_value,
		       const double y_max_value,
		       const int num_x,
		       const int num_y,
		       const double dx,
		       const double dy,
		       const double dt,
		       const double alpha );

    // Destructor.
    ~DiffusionOperator();

    // Get the operator.
    const Teuchos::RCP<Epetra_CrsMatrix>& getCrsMatrix() const
    { return d_matrix; }

  private:

    // Build the five point stencil diffusion operator.
    void build_five_point_stencil( const int num_x, 
				   const int num_y,
				   const double dx,
				   const double dy,
				   const double dt,
				   const double alpha );

    // Build the nine point stencil diffusion operator.
    void build_nine_point_stencil( const int num_x, 
				   const int num_y,
				   const double dx,
				   const double dy,
				   const double dt,
				   const double alpha );
};

} // end namespace HMCSA

#endif // end HMCSA_DIFFUSIONOPERATOR_HPP

//---------------------------------------------------------------------------//
// end DiffusionOperator.hpp
//---------------------------------------------------------------------------//

