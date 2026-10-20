# ForgeRT command-line tools (Phase 9). CPU-only: they link forgert_import,
# never forgert_cuda (see src/CMakeLists.txt).

add_executable(fgx_dump fgx_dump.cpp)
target_link_libraries(fgx_dump PRIVATE forgert_import)

message(STATUS "ForgeRT apps configured")