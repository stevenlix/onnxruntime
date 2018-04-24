#include "core/providers/cpu/tensor/identity_op.h"

namespace Lotus {
//copying reshape kernel
REGISTER_KERNEL(KernelDefBuilder("Dropout")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                IdentityOp<float>);

REGISTER_KERNEL(KernelDefBuilder("Identity")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .Alias(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                IdentityOp<float>);

}  // namespace Lotus
