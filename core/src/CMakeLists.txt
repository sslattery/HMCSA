# CMakeLists.txt for src

INCLUDE(TribitsLibraryMacros)

#
# A) Package-specific configuration options
#

#
# B) Define the header and source files (and include directories)
#

SET(HEADERS "")
SET(SOURCES "")

# Set headers (for IDEs and installing)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  AdjointMC.hpp
  DiffusionOperator.hpp
  DirectMC.hpp
  Exception.hpp
  HMCSATypes.hpp
  JacobiPreconditioner.hpp
  MCSA.hpp
  OperatorTools.hpp
  SequentialMC.hpp
  VtkWriter.hpp
  )

# Set sources

APPEND_SET(SOURCES
  AdjointMC.cpp
  DiffusionOperator.cpp
  DirectMC.cpp
  Exception.cpp
  JacobiPreconditioner.cpp
  MCSA.cpp
  OperatorTools.cpp
  SequentialMC.cpp
  VtkWriter.cpp
  )

# Set addition headers and source based on package-specific
# configuration options

IF(${PACKAGE_NAME}_ENABLE_EXTENDED)
APPEND_SET(HEADERS
    )

  APPEND_SET(SOURCES
    )

ENDIF()

#
# C) Define the targets for install
#

TRIBITS_ADD_LIBRARY(
  hmcsa_utils
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )

INSTALL(FILES 
  ${HEADERS} 
  DESTINATION 
  ${CMAKE_INSTALL_PREFIX}/include/hmcsa/)