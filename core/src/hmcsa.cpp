//---------------------------------------------------------------------------//
// \file hmcsa.cpp
// \author Stuart R. Slattery
// \brief HMCSA executable.
//---------------------------------------------------------------------------//

#include <iostream>

#include "HelmholtzOperator.hpp"
#include "MCSA.hpp"
#include "TimeIntegrator.hpp"
#include "VtkWriter.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_Vector.h>
#include <Epetra_LinearProblem.h>

int main( int argc, void** argv )
{
    // Problem parameters.
    Teuchos::ParameterList parameter_list("problem1");

    // Grid.
    parameter_list.set( "X Min", 0.0 );
    parameter_list.set( "X Max", 1.0 );
    parameter_list.set( "Y Min", 0.0 );
    parameter_list.set( "Y Max", 1.0 );
    parameter_list.set( "Grid Size", 0.01 );

    // Time Step.
    parameter_list.set( "Time Step Size", 0.001 );
    parameter_list.set( "Final Time", 0.01 );

    // Dirichlet Conditions.
    parameter_list.set( "X Min Boundary", 0.0 );
    parameter_list.set( "X Max Boundary", 1.0 );
    parameter_list.set( "Y_Min Boundary", 0.0 );
    parameter_list.set( "Y_Max Boundary", 1.0 );

    // Initial Conditions.
    parameter_list.set( "Initial Value", 10.0 );    

    // Solver.
    parameter_list.set( "Solver", "MCSA" );
    parameter_list.set( "Histories", 1000 );
    parameter_list.set( "Cutoff Weight", 1.0E-12 );
    parameter_list.set( "Max Iterations", 100 );
    parameter_list.set( "Tolerance", 1.0e-8 );

    // Epetra map setup.
    Epetra_Map map;

    // Create the solution vector and assign initial conditons.
    Teuchos::ArrayRCP<double> u_array;
    Epetra_Vector u( View, map, u.get() );

    // Create the right hand side.
    Teuchos::ArrayRCP<double> b_array;
    Epetra_Vector b( View, map, b.get() );
    
    // Create the operator.
    HMCSA::HelmholtzOperator A( parameter_list );

    // Create the linear problem.
    Epetra_LinearProblem linear_problem( A->getCrsMatrix(), u, b );

    // Create a solver.
    HMCSA::MCSA linear_solver;

    // Create a time integrator.
    HMCSA::TimeIntegrator time_integrator( parameter_list, 
					   linear_solver, 
					   linear_problem );

    // Integrate.
    time_integrator.integrate();

    // Write to vtk.
    VtkWriter vtk_writer( parameter_list );
    vtk_writer.addVector( u_array, "u" );
    vtk_writer.write();

    return 0;
}
