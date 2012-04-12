#
# See documentation in Trilinos preCopyrightTrilinos/ExtraExternalRepositories.cmake
#

INCLUDE(TribitsListHelpers)

SET( HMCSA_PACKAGES_AND_DIRS_AND_CLASSIFICATIONS
  HMCSA         .     SS
  )

PACKAGE_DISABLE_ON_PLATFORMS(HMCSA Windows)
