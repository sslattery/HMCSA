//---------------------------------------------------------------------------//
// \file DiffusionOperator.cpp
// \author Stuart R. Slattery
// \brief Diffusion operator definition.
//---------------------------------------------------------------------------//

#include "DiffusionOperator.cpp"

namespace HMCSA
{
/*! 
 * \brief Constructor.
 */
DiffusionOperator::DiffusionOperator( const int x_min_type,
				      const int x_max_type,
				      const int y_min_type,
				      const int y_max_type,
				      const double x_min_value,
				      const double x_max_value,
				      const double y_min_value,
				      const double y_max_value,
				      const int size_x,
				      const int size_y )
    : d_x_min_type( x_min_type )
    , d_x_max_type( x_max_type )
    , d_y_min_type( y_min_type )
    , d_y_max_type( y_max_type )
    , d_x_min_value( x_min_value )
    , d_x_max_value( x_max_value )
    , d_y_min_value( y_min_value )
    , d_y_max_value( y_max_value )
{
    d_matrix = build_diffusion_operator( size_x, size_y );
}

/*!
 * \brief Destructor.
 */
DiffusionOperator::~DiffusionOperator()
{ /* ... */ }

/*!
 * \brief Build the diffusion operator.
 */
CrsMatrix DiffusionOperator::build_diffusion_operator( const int size_x, 
						       const int size_y )
{

}

} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end DiffusionOperator.cpp
//---------------------------------------------------------------------------//

