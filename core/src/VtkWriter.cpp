//---------------------------------------------------------------------------//
// \file VtkWriter.cpp
// \author Stuart R. Slattery
// \brief VtkWriter definition.
//---------------------------------------------------------------------------//

#include <string>
#include <sstream>
#include <cassert>

#include "VtkWriter.hpp"

#include <MBCore.hpp>

namespace HMCSA
{

/*!
 * \brief Constructor.
 */
VtkWriter::VtkWriter( const double x_min,
		      const double x_max,
		      const double y_min,
		      const double y_max,
		      const double dx,
		      const double dy,
		      const int num_x,
		      const int num_y )
{
    // Create core instance.
    d_MBI = Teuchos::rcp( new moab::Core() );

    // Error code.
    moab::ErrorCode rval;

    // Create the vertices.
    std::vector<double> coords;
    unsigned int num_verts = num_x*num_y;
    coords.resize(num_verts*3);

    std::vector<double> i_arr;
    for (int i = 0; i < num_x; ++i) 
    {
	i_arr.push_back( x_min + i*dx );
    }

    std::vector<double> j_arr;
    for (int j = 0; j < num_y; ++j) 
    {
	j_arr.push_back( y_min + j*dy );
    }

    int idx;
    for (int j = 0; j < num_y; ++j) 
    {
	for (int i = 0; i < num_x; ++i) 
	{
	    idx = i + num_x*j;
	    coords[3*idx] = i_arr[i];
	    coords[3*idx + 1] = j_arr[j];
	    coords[3*idx + 2] = 0.0;
	}
    }

    rval = d_MBI->create_vertices( &coords[0], num_verts, d_vtx_range );
    assert( moab::MB_SUCCESS == rval );

    // create the quads
    moab::EntityHandle conn[4];
    for (int j = 0; j < num_y - 1; ++j) 
    {
	for (int i = 0; i < num_x - 1; ++i) 
	{
	    idx = i + (num_x)*j;
	    conn[0] = d_vtx_range[idx];
	    conn[1] = d_vtx_range[idx + 1];
	    conn[2] = d_vtx_range[idx + num_x + 1];
	    conn[3] = d_vtx_range[idx + num_x];

	    moab::EntityHandle this_quad;
	    rval = d_MBI->create_element(moab::MBQUAD, conn, 4, this_quad);
	    assert( moab::MB_SUCCESS == rval );
	}
    }

    // Create the tag.
    moab::Tag u_tag;
    rval = d_MBI->tag_create("u", 
			   sizeof(double),
			   moab::MB_TAG_DENSE,
			   moab::MB_TYPE_DOUBLE,
			   u_tag, 
			   0);
    assert(moab::MB_SUCCESS == rval);
}


// destructor
VtkWriter::~VtkWriter()
{ /* ... */ }

/*!
 * \brief Write data to a mesh file. 
 */
void VtkWriter::write_vector( const std::vector<double> &u,
			      const std::string &name )
{
    // Error value.
    moab::ErrorCode rval;

    // Tag the vertices.
    moab::Tag u_tag;
    rval = d_MBI->tag_get_handle( "u", 
				  1,
				  moab::MB_TYPE_DOUBLE,
				  u_tag );
    assert(moab::MB_SUCCESS == rval);

    std::string filename = "time" + name + ".vtk";

    rval = d_MBI->tag_set_data( u_tag, d_vtx_range, &u[0] );
    assert( moab::MB_SUCCESS == rval );

    rval = d_MBI->write_mesh( &filename[0] );
    assert( moab::MB_SUCCESS == rval );
}

} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end VtkWriter.cpp
//---------------------------------------------------------------------------//

