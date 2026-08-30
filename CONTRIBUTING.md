# Contributing to ForgeRT

Thank you for your interest in contributing to ForgeRT! This is an educational/research project focused on resource-aware neural network inference.

## How to Contribute

### Reporting Issues

When reporting bugs or suggesting features:

1. **Search existing issues** first
2. **Provide detailed information**:
   - Operating system and version
   - CUDA Toolkit version
   - GPU model and compute capability
   - CMake version
   - Compiler version
   - Full error messages
   - Minimal reproducible example

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes**
4. **Test thoroughly**:
   - All existing tests pass
   - New functionality has tests
   - Benchmarks run successfully
5. **Document your changes**:
   - Update relevant documentation
   - Add comments explaining complex logic
   - Update CHANGELOG.md if applicable
6. **Commit with clear messages**
7. **Push to your fork**
8. **Open a pull request**

### Code Review Process

All submissions require review. We'll look for:

- **Correctness**: Does it work as intended?
- **Performance**: Is there measurement to back claims?
- **Code quality**: Is it readable and maintainable?
- **Testing**: Are edge cases covered?
- **Documentation**: Can others understand it?

## Development Guidelines

### Before Starting

- Discuss major changes in an issue first
- Check that your contribution aligns with project goals
- Ensure you can test on target hardware (or clearly state limitations)

### Code Standards

- Follow existing code style
- Use `clang-format` for C++ files
- Write clear, self-documenting code
- Add comments for complex algorithms
- Include references for mathematical operations

### Testing Requirements

- **Unit tests** for new functionality
- **Integration tests** for multi-component features
- **Benchmark** for performance-critical code
- All tests must pass on Windows with MSVC and CUDA

### Performance Claims

**Never claim a performance improvement without measurement.**

Required for performance-related PRs:
1. Baseline measurement
2. Optimized measurement
3. Statistical analysis (multiple runs)
4. Hardware details
5. Profiling results explaining why it's faster

### Documentation

Update documentation for:
- New public APIs
- Architectural changes
- Build system modifications
- New dependencies
- Benchmark procedures

## What We're Looking For

### High Priority

- Core tensor operations
- CUDA kernel implementations
- Memory management improvements
- Scheduler enhancements
- Benchmark infrastructure
- Documentation improvements

### Nice to Have

- Additional operator implementations
- Graph optimization passes
- Visualization tools
- Python tooling
- Examples and tutorials

### Not Accepting

- Dependencies on heavy frameworks (PyTorch, TensorFlow as runtime deps)
- Support for non-target hardware without clear abstraction
- Features that hide execution decisions
- Optimizations without measurements
- Automatic GPU offloading without cost analysis

## Questions?

Feel free to:
- Open an issue for discussion
- Ask in pull request comments
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Remember**: ForgeRT is about understanding and demonstrating resource-aware execution, not building the fastest possible runtime. Clarity and measurability are as important as performance.
