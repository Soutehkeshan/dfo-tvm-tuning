## 2026-11-02

- **TVM build (Claude Code):** Used Claude Code to build TVM v0.27.0.post1 from source (CMake/LLVM config), diagnose a Ninja/Make mismatch, and fix library discovery via a .pth file. I checked the result by importing TVM and confirming the version and runtime library path.
- **Dependency check (Claude):** Claude suggested confirming the tvm-ffi wheel matches TVM's pinned submodule; I ran the check and both were v0.1.14.post0.
- **Smoke test (Claude):** Used Claude to write a setup smoke test (scripts/smoke_test.py) that compiles and runs a vector-add kernel via LLVM; I ran it and it passed.
- **README (Claude):** Claude suggested recording the build environment (LLVM, Python, macOS) in the README for reproducibility; I generated the details from my machine.
