//---------------------------------------------------------------------------//
// Visualize.hpp
// Visualize class definition
//---------------------------------------------------------------------------//

#ifndef __VISUALIZEHPP__
#define __VISUALIZEHPP__

#include "Matrix.hpp"
#include "Properties.hpp"
#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"
#include <vector>

//---------------------------------------------------------------------------//

class Visualize
{
public:

  // constructor
  Visualize(std::vector<Matrix*> results, Properties *props);

  // destructor
  ~Visualize();

}; // end class visualize

#endif // __VISUALIZEHPP__

//---------------------------------------------------------------------------//
// end Visualize.hpp
//---------------------------------------------------------------------------//

