//---------------------------------------------------------------------------//
// \file DiffusionOperator.cpp
// \author Stuart R. Slattery
// \brief Second order diffusion operator definition.
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
DiffusionOperator::DiffusionOperator( const int order,
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
    if ( order == 2 )
    {
	build_five_point_stencil( num_x, num_y, dx, dy, dt, alpha );
    }
    else if ( order == 4 )
    {
	build_nine_point_stencil( num_x, num_y, dx, dy, dt, alpha );
    }
}

/*!
 * \brief Destructor.
 */
DiffusionOperator::~DiffusionOperator()
{ /* ... */ }

/*!
 * \brief Build the five point stencil diffusion operator.
 */
void DiffusionOperator::build_five_point_stencil( const int num_x, 
						  const int num_y,
						  const double dx,
						  const double dy,
						  const double dt,
						  const double alpha )
{
    int N = num_x*num_y;

    Epetra_SerialComm comm;
    Epetra_Map map( N, 0, comm );

    std::vector<int> entries_per_row( N, 5 );

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
    d_matrix->OptimizeStorage();
}

/*!
 * \brief Build the nine point stencil diffusion operator.
 */
void DiffusionOperator::build_nine_point_stencil( const int num_x, 
						  const int num_y,
						  const double dx,
						  const double dy,
						  const double dt,
						  const double alpha )
{
    int N = num_x*num_y;

    Epetra_SerialComm comm;
    Epetra_Map map( N, 0, comm );

    std::vector<int> entries_per_row( N, 9 );

    d_matrix = Teuchos::rcp( 
	new Epetra_CrsMatrix( Copy, map, &entries_per_row[0] ) );

    double diag = 1.0 + 15.0*dt*alpha*( 1/(dx*dx) + 1/(dy*dy) )/6.0;
    double i_minus1 = -4.0*dt*alpha/(3.0*dx*dx);
    double i_plus1 = -4.0*dt*alpha/(3.0*dx*dx);
    double j_minus1 = -4.0*dt*alpha/(3.0*dy*dy);
    double j_plus1 = -4.0*dt*alpha/(3.0*dy*dy);
    double i_minus2 = dt*alpha/(12.0*dx*dx);
    double i_plus2 = dt*alpha/(12.0*dx*dx);
    double j_minus2 = dt*alpha/(12.0*dy*dy);
    double j_plus2 = dt*alpha/(12.0*dy*dy);

    int idx;
    int idx_iminus1;
    int idx_iplus1;
    int idx_jminus1;
    int idx_jplus1;
    int idx_iminus2;
    int idx_iplus2;
    int idx_jminus2;
    int idx_jplus2;

    double one = 1.0;

    double diag_2 = 1.0 + 2*dt*alpha*( 1/(dx*dx) + 1/(dy*dy) );
    double i_minus = -dt*alpha/(dx*dx);
    double i_plus = -dt*alpha/(dx*dx);
    double j_minus = -dt*alpha/(dy*dy);
    double j_plus = -dt*alpha/(dy*dy);

    int idx_iminus;
    int idx_iplus;
    int idx_jminus;
    int idx_jplus;

    // Fill in with a 5 point stencil to hit the near boundary points that
    // can't be used with the 9 point. Just being lazy here.
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
	    d_matrix->InsertGlobalValues( idx, 1, &diag_2,  &idx        );
	    d_matrix->InsertGlobalValues( idx, 1, &i_plus,  &idx_iplus  );
	    d_matrix->InsertGlobalValues( idx, 1, &j_plus,  &idx_jplus  );
	}
    }

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
    for ( int i = 2; i < num_x-2; ++i )
    {
	for ( int j = 2; j < num_y-2; ++j )
	{
	    idx = i + j*num_x;
	    idx_iminus1 = (i-1) + j*num_x;
	    idx_iplus1 = (i+1) + j*num_x;
	    idx_jminus1 = i + (j-1)*num_x;
	    idx_jplus1 = i + (j+1)*num_x;
	    idx_iminus2 = (i-2) + j*num_x;
	    idx_iplus2 = (i+2) + j*num_x;
	    idx_jminus2 = i + (j-2)*num_x;
	    idx_jplus2 = i + (j+2)*num_x;

	    d_matrix->InsertGlobalValues( idx, 1, &j_minus2, &idx_jminus2 );
	    d_matrix->InsertGlobalValues( idx, 1, &i_minus2, &idx_iminus2 );
	    d_matrix->InsertGlobalValues( idx, 1, &j_minus1, &idx_jminus1 );
	    d_matrix->InsertGlobalValues( idx, 1, &i_minus1, &idx_iminus1 );
	    d_matrix->InsertGlobalValues( idx, 1, &diag,     &idx         );
	    d_matrix->InsertGlobalValues( idx, 1, &i_plus1,  &idx_iplus1  );
	    d_matrix->InsertGlobalValues( idx, 1, &j_plus1,  &idx_jplus1  );
	    d_matrix->InsertGlobalValues( idx, 1, &i_plus2,  &idx_iplus2  );
	    d_matrix->InsertGlobalValues( idx, 1, &j_plus2,  &idx_jplus2  );
	}
    }

    d_matrix->FillComplete();
    d_matrix->OptimizeStorage();
}

} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end DiffusionOperator.cpp
//---------------------------------------------------------------------------//

