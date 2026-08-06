include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_tridiagonal_batch test_tridiagonal_batch.cpp)
target_link_libraries(test_tridiagonal_batch PRIVATE dtdma Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(test_tridiagonal_batch)

add_executable(test_thomas_solver test_thomas_solver.cpp)
target_link_libraries(test_thomas_solver PRIVATE dtdma Catch2::Catch2WithMain)
catch_discover_tests(test_thomas_solver)

add_executable(test_batched_thomas_solver test_batched_thomas_solver.cpp)
target_link_libraries(test_batched_thomas_solver
                      PRIVATE dtdma Catch2::Catch2WithMain)
catch_discover_tests(test_batched_thomas_solver)

add_executable(test_forward_coefficient_preparation
               test_forward_coefficient_preparation.cpp)
target_link_libraries(test_forward_coefficient_preparation
                      PRIVATE dtdma Catch2::Catch2WithMain)
catch_discover_tests(test_forward_coefficient_preparation)

add_executable(test_forward_rhs_reduction test_forward_rhs_reduction.cpp)
target_link_libraries(test_forward_rhs_reduction
                      PRIVATE dtdma Catch2::Catch2WithMain)
catch_discover_tests(test_forward_rhs_reduction)

add_executable(test_single_partition_solve test_single_partition_solve.cpp)
target_link_libraries(test_single_partition_solve
                      PRIVATE dtdma Catch2::Catch2WithMain)
catch_discover_tests(test_single_partition_solve)

add_executable(test_uniform_virtual_solve test_uniform_virtual_solve.cpp)
target_link_libraries(test_uniform_virtual_solve
                      PRIVATE dtdma Catch2::Catch2WithMain)
catch_discover_tests(test_uniform_virtual_solve)
