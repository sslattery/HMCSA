//---------------------------------------------------------------------------//
// \file DiffusionOperator.cpp
// \author Stuart R. Slattery
// \brief Diffusion operator definition.
//---------------------------------------------------------------------------//

#include <vector>

#include "DiffusionOperator.hpp"

#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>

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
				      const int num_x,
				      const int num_y,
                                      const double dx,
                                      const double dy,
				      const double dt,
				      const double alpha )
    : d_x_min_type( x_min_type )
    , d_x_max_type( x_max_type )
    , d_y_min_type( y_min_type )
    , d_y_max_type( y_max_type )
    , d_x_min_value( x_min_value )
    , d_x_max_value( x_max_value )
    , d_y_min_value( y_min_value )
    , d_y_max_value( y_max_value )
{
    build_diffusion_operator( num_x, num_y, dx, dy, dt, alpha );
}

/*!
 * \brief Destructor.
 */
DiffusionOperator::~DiffusionOperator()
{ /* ... */ }

/*!
 * \brief Build the diffusion operator.
 */
void DiffusionOperator::build_diffusion_operator( const int num_x, 
						  const int num_y,
						  const double dx,
						  const double dy,
						  const double dt,
						  const double alpha )
{
    int N = num_x*num_y;
    std::vector<int> entries_per_row( N, 5 );

    Epetra_SerialComm comm;
    Epetra_Map map( N, 0, comm );

    d_matrix = Teuchos::rcp( 
	new Epetra_CrsMatrix( Copy, map, &entries_per_row[0] ) );

    double diag = 1.0 + 2*dt*alpha*( 1/(dx*dx) + 1/(dy*dy) );
    double i_minus = -dt*alpha/(dx*dx);
    double i_plus = -dt*alpha/(dx*dx);
    double j_minus = -dt*alpha/(dy*dy);
    double j_plus = -dt*alpha/(dy*dy);

    int idx;
    int idx_iminus;
    int idx_iplus;
    int idx_jminus;
    int idx_jplus;
    double one = 1.0;

    // Min X boundary Dirichlet.
    for ( int j = 1; j < num_y-1; ++j )
    {
	int i = 0;
	idx = i + j*num_x;
	d_matrix->InsertGlobalValues( idx, 1, &one, &idx );
    }

    // Max X boundary Dirichlet.
    for ( int j = 1; j < num_y-1; ++j )
    {
	int i = num_x-1;
	idx = i + j*num_x;
	d_matrix->InsertGlobalValues( idx, 1, &one, &idx );
    }

    // Min Y boundary Dirichlet.
    for ( int i = 0; i < num_x; ++i )
    {
	int j = 0;
	idx = i + j*num_x;
	d_matrix->InsertGlobalValues( idx, 1, &one, &idx );
    }

    // Max Y boundary Dirichlet.
    for ( int i = 0; i < num_x; ++i )
    {
	int j = num_y-1;
	idx = i + j*num_x;
	d_matrix->InsertGlobalValues( idx, 1, &one, &idx );
    }

    // Central grid points.
    for ( int i = 1; i < num_x-1; ++i )
    {
	for ( int j = 1; j < num_y-1; ++j )
	{
	    idx = i + j*num_x;
	    idx_iminus = (i-1) + j*num_x;
	    idx_iplus = (i+1) + j*num_x;
	    idx_jminus = i + (j-1)*num_x;
	    idx_jplus = i + (j+1)*num_x;
	    
	    d_matrix->InsertGlobalValues( idx, 1, &j_minus, &idx_jminus );
	    d_matrix->InsertGlobalValues( idx, 1, &i_minus, &idx_iminus );
	    d_matrix->InsertGlobalValues( idx, 1, &diag,    &idx        );
	    d_matrix->InsertGlobalValues( idx, 1, &i_plus,  &idx_iplus  );
	    d_matrix->InsertGlobalValues( idx, 1, &j_plus,  &idx_jplus  );
	}
    }

    d_matrix->FillComplete();
}

} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end DiffusionOperator.cpp
//---------------------------------------------------------------------------//

