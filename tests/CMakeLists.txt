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
