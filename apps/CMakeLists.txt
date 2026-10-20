# ForgeRT command-line tools (Phase 9). CPU-only: they link forgert_import,
# never forgert_cuda (see src/CMakeLists.txt).

# Canonical manifest of a .fgx file (used by the cross-language check)
add_executable(fgx_dump fgx_dump.cpp)
target_link_libraries(fgx_dump PRIVATE forgert_import)

# Run a single-input .fgx model on CPU: raw float32 in, raw float32 out
add_executable(forgert_run forgert_run.cpp)
target_link_libraries(forgert_run PRIVATE forgert_import)

message(STATUS "ForgeRT apps configured")