# Security Policy

ForgeRT is an educational/research project. It does not handle user data,
network connections, or production workloads. The attack surface is limited
to local file I/O (model weight files) and CPU/GPU computation.

## Supported Versions

Only the current main branch is maintained. No backports or patches are
issued for older commits.

## Reporting a Vulnerability

If you find a security issue (for example, a path-traversal bug in the weight
loader, or unsafe use of user-supplied data), please report it privately rather
than opening a public issue.

**How to report:**

1. Open a [GitHub Security Advisory](https://github.com/psamuelvijay/ForgeRT/security/advisories/new)
   on this repository, or
2. Contact the maintainer directly through the GitHub profile at
   https://github.com/psamuelvijay

Include a short description of the issue and steps to reproduce it.

## What to Expect

This is a solo research project maintained in spare time. There are no
guaranteed response timelines or SLAs. Straightforward fixes will be
addressed as quickly as possible; complex issues may take longer.

Reports will be acknowledged and kept confidential until a fix is available.

## Scope

In scope:
- Memory safety bugs in C++ tensor or graph code
- Path traversal or arbitrary file read via the .fgwt weight loader
- Undefined behaviour that could be triggered by crafted weight files

Out of scope:
- Denial-of-service from large weight files (no resource limits are claimed)
- Issues in third-party dependencies (PyTorch, CUDA Toolkit, MSVC)
- Theoretical issues with no practical impact on this project
