//---------------------------------------------------------------------------//
// Visualize.cpp
// Visualize member defintions
//---------------------------------------------------------------------------//

#include "Visualize.hpp"
#include "Matrix.hpp"
#include "Properties.hpp"
#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"
#include <vector>
#include <string>
#include <sstream>
#include <cassert>

//---------------------------------------------------------------------------//

// constructor
Visualize::Visualize(std::vector<Matrix*> results, Properties *props)
{

  // create core instance
  moab::Interface *MBI = new moab::Core();

  // error value
  moab::ErrorCode rval;

  // create the vertices
  std::vector<double> coords;
  unsigned int num_i = props->x_steps;
  unsigned int num_j = props->y_steps;
  unsigned int num_verts = num_i*num_j;
  coords.resize(num_verts*3);

  std::vector<double> i_arr;
  unsigned int x;
  for (x = 0; x != num_i; x++) {
    i_arr.push_back(x*props->hx);
  }

  std::vector<double> j_arr;
  unsigned int y;
  for (y = 0; y != num_j; y++) {
    j_arr.push_back(y*props->hy);
  }

  unsigned int ival, jval, idx;
  for (jval = 0; jval != num_j; jval++) {
    for (ival = 0; ival != num_i; ival++) {
      idx = ival + num_i*jval;
      coords[3*idx] = i_arr[ival];
      coords[3*idx + 1] = j_arr[jval];
      coords[3*idx + 2] = 0.0;
    }
  }

  moab::Range vtx_range;
  rval = MBI->create_vertices(&coords[0], num_verts, vtx_range);
  assert(moab::MBSUCCESS == rval);

  // create the quads
  moab::EntityHandle conn[4];
  unsigned int iq, jq;
  for (jq = 0; jq != num_j - 1; jq++) {
    for (iq = 0; iq != num_i - 1; iq++) {
      idx = iq + (num_i)*jq;
      conn[0] = vtx_range[idx];
      conn[1] = vtx_range[idx + 1];
      conn[2] = vtx_range[idx + num_i + 1];
      conn[3] = vtx_range[idx + num_i];


      moab::EntityHandle this_quad;
      rval = MBI->create_element(moab::MBQUAD, conn, 4, this_quad);
      assert(moab::MBSUCCESS == rval);
    }
  }

  // tag the vertices
  moab::Tag temp;
  rval = MBI->tag_create("TEMPERATURE", 
                         sizeof(double),
                         moab::MB_TAG_DENSE,
                         moab::MB_TYPE_DOUBLE,
                         temp, 
                         0);
  assert(moab::MBSUCCESS == rval);

  unsigned int t;
  std::stringstream convert;
  std::string filename;
  for (t = 0; t != results.size(); t++) {

    std::stringstream convert;
    convert << t;
    filename = "time" + convert.str() + ".vtk";

    rval = MBI->tag_set_data(temp, vtx_range, results[t]->elements);
    assert(moab::MBSUCCESS == rval);

    rval = MBI->write_mesh(&filename[0]);
    assert(moab::MBSUCCESS == rval);
  }
}

// destructor
Visualize::~Visualize()
{
}

//---------------------------------------------------------------------------//
// end Visualize.cpp
//---------------------------------------------------------------------------//

