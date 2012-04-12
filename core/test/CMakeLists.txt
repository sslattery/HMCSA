INCLUDE(TribitsAddExecutableAndTest)
INCLUDE(TribitsAddAdvancedTest)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Epetra_test
  SOURCES tstEpetra.cpp ${TEUCHOS_STD_UNIT_TEST_MAIN}
  COMM serial
  STANDARD_PASS_OUTPUT
  )
