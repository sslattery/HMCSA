//---------------------------------------------------------------------------//
// \file VtkWriter.hpp
// \author Stuart Slattery
// \brief VtkWriter class declaration.
//---------------------------------------------------------------------------//

#ifndef HMCSA_VTKWRITER_HPP
#define HMCSA_VTKWRITER_HPP

#include <vector>

#include <Teuchos_RCP.hpp>

#include <MBInterface.hpp>
#include <MBRange.hpp>

namespace HMCSA
{

class VtkWriter
{
  private:

    // MOAB instance.
    Teuchos::RCP<moab::Interface> d_MBI;

    // Vertex range.
    moab::Range d_vtx_range;

  public:

    // Constructor.
    VtkWriter( const double x_min,
	       const double x_max,
	       const double y_min,
	       const double y_max,
	       const double dx,
	       const double dy,
	       const int num_x,
	       const int num_y );

    // Destructor.
    ~VtkWriter();

    // Write data to a mesh file. 
    void write_vector( const std::vector<double> &u, 
		       const std::string &name );
}; 

} // end namespace HMCSA

#endif // end HMCSA_VTKWRITER_HPP

//---------------------------------------------------------------------------//
// end VtkWriter.hpp
//---------------------------------------------------------------------------//

