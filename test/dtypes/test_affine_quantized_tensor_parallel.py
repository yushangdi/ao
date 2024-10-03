from torchao.testing.utils import copy_tests, TorchAOTensorParallelTestCase
from torch.testing._internal.common_utils import run_tests
from torchao.quantization import int8_weight_only, float8_weight_only

# class TestAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
#     pass

class TestAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
    QUANT_METHOD_FN = staticmethod(float8_weight_only)

print('Copy test started...')
copy_tests(TorchAOTensorParallelTestCase, TestAffineQuantizedTensorParallel, "fp8wo_tp")
print('Copy test finished')

if __name__ == "__main__":
    print("Running TestAffineQuantizedTensorParallel")
    run_tests()
