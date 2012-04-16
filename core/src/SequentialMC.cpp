//---------------------------------------------------------------------------//
// \file SequentialMC.cpp
// \author Stuart R. Slattery
// \brief Monte Carlo Synthetic Acceleration solver declaration.
//---------------------------------------------------------------------------//

#include "SequentialMC.hpp"
#include "AdjointMC.hpp"

#include <Epetra_Map.h>
#include <Epetra_Vector.h>

namespace HMCSA
{
/*! 
 * \brief Constructor.
 */
SequentialMC::SequentialMC( Epetra_LinearProblem *linear_problem )
    : d_linear_problem( linear_problem )
    , d_num_iters( 0 )
{ /* ... */ }

/*!
 * \brief Destructor.
 */
SequentialMC::~SequentialMC()
{ /* ... */ }
 
/*!
 * \brief Solve.
 */
void SequentialMC::iterate( const int max_iters,
			    const double tolerance,
			    const int num_histories,
			    const double weight_cutoff )
{
    Epetra_CrsMatrix *A = 
	dynamic_cast<Epetra_CrsMatrix*>( d_linear_problem->GetMatrix() );
    Epetra_Vector *x = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetLHS() );
    const Epetra_Vector *b = 
	dynamic_cast<Epetra_Vector*>( d_linear_problem->GetRHS() );

    Epetra_Map row_map = A->RowMap();
    Epetra_Vector delta_x( row_map );
    Epetra_Vector residual( row_map );
    Epetra_LinearProblem residual_problem( A, &delta_x, &residual );
    AdjointMC mc_solver( &residual_problem );

    Epetra_Vector temp_vec( row_map );
    int N = A->NumGlobalRows();

    d_num_iters = 0;
    double residual_norm = 1.0;
    double b_norm;
    b->NormInf( &b_norm );
    double conv_crit = b_norm*tolerance;
    while ( residual_norm > conv_crit && d_num_iters < max_iters )
    {
	A->Apply( *x, temp_vec );
	for ( int i = 0; i < N; ++i )
	{
	    residual[i] = (*b)[i] - temp_vec[i];
	    std::cout << residual[i] << std::endl;
	}
	std::cout << "---" << std::endl;
	mc_solver.walk( num_histories, weight_cutoff );

	for ( int i = 0; i < N; ++i )
	{
	    (*x)[i] += delta_x[i];
	}

	residual.NormInf( &residual_norm );
	++d_num_iters;
    }
}

} // end namespace HMCSA

//---------------------------------------------------------------------------//
// end SequentialMC.cpp
//---------------------------------------------------------------------------//

