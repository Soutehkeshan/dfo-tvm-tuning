"""Setup check: TVM compiles a kernel via LLVM and runs it correctly on this machine."""
import numpy as np
import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), name="A", dtype="float32")
B = te.placeholder((n,), name="B", dtype="float32")
C = te.compute((n,), lambda i: A[i] + B[i], name="C")
func = te.create_prim_func([A, B, C])

build = getattr(tvm, "compile", None) or tvm.build
lib = build(func, target="llvm")

dev = tvm.cpu()
to_dev = getattr(tvm.runtime, "tensor", None) or tvm.nd.array
a_np = np.random.rand(n).astype("float32")
b_np = np.random.rand(n).astype("float32")
a, b = to_dev(a_np, dev), to_dev(b_np, dev)
c = to_dev(np.zeros(n, dtype="float32"), dev)

lib(a, b, c)
np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-5)
print("OK: TVM", tvm.__version__, "compiled and ran via LLVM")
