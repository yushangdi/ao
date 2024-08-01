import torch
from typing import Tuple, Optional
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    fake_quantize_affine,
    ZeroPointDomain,
    MappingType,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.utils import (
    _implements,
    _dispatch__torch_function__,
    _dispatch__torch_dispatch__,
)
from .utils import _GenericFakeQuantize

aten = torch.ops.aten

class AffineFakeQuantizedTensor(torch.Tensor):
    """
    Affine fake quantized tensor subclass. Affine quantization means we quantize the floating point tensor
    with an affine transformation:
       quantized_tensor = float_tensor / scale + zero_point

    Fake quantization refers to performing the quantization math without actually casting the floating point
    tensor into lower bit-width dtypes. It is commonly used for quantization-aware training (QAT).

    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.

    fields:
      fq_data (torch.Tensor): actual tensor holding the fake quantized values
      block_size (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
         e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      shape (torch.Size): the shape for the Tensor
      quant_min (Optional[int]): minimum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      quant_max (Optional[int]): maximum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
            value during quantization
        default is ZeroPointDomain.INT
    """

    @staticmethod
    def __new__(
        cls,
        fq_data: torch.Tensor,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype: torch.dtype = None,
    ):
        kwargs = {}
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = True
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        fq_data: torch.Tensor,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype: torch.dtype = None,
    ):
        self.fq_data = fq_data
        self.block_size = block_size
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.zero_point_domain = zero_point_domain

    def __tensor_flatten__(self):
        return ["fq_data"], [self.block_size, self.shape, self.quant_min, self.quant_max, self.zero_point_domain, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride,
    ):
        fq_data = tensor_data_dict["fq_data"]
        block_size, shape, quant_min, quant_max, zero_point_domain, dtype = tensor_attributes
        return cls(
            fq_data,
            block_size,
            shape if outer_size is None else outer_size,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=dtype,
        )

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        mapping_type: MappingType,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: Optional[int] = None,
        quant_max: Optional[int]  = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    ):
        original_shape = input_float.shape
        scale, zero_point = choose_qparams_affine(
            input_float,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
            preserve_zero,
            zero_point_domain,
        )
        fq_data = _GenericFakeQuantize.apply(
            input_float,
            block_size,
            scale,
            zero_point,
            quant_min,
            quant_max,
            zero_point_domain,
        )
        return cls(
            fq_data,
            block_size,
            original_shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype
        )

    def _get_to_kwargs(self, *args, **kwargs):
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        memory_format = ( 
            memory_format if memory_format is not None else torch.preserve_format
        )   
        kwargs = { 
            "device": device,
            "dtype": dtype,
            "memory_format": memory_format,
        }   
        return kwargs

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        # not supported yet
        kwargs.pop("memory_format")
        return self.__class__(
            self.fq_data.to(device),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.fq_data),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            dtype=self.dtype,
        )

    implements = classmethod(_implements)
    __torch_function__ = classmethod(_dispatch__torch_function__)
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)

implements = AffineFakeQuantizedTensor.implements


@implements(torch.nn.functional.linear)
def _(func, types, *args, **kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, AffineFakeQuantizedTensor):
        weight_tensor = weight_tensor.fq_data
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

@implements([aten.mm.default, aten.addmm.default])
def _(func, types, *args, **kwargs):
    if func == aten.addmm.default:
        bias = args[0]
        input_index = 1
    else:
        bias = None
        input_index = 0
    input_tensor = args[input_index]
    weight_tensor = args[input_index + 1]
    if isinstance(weight_tensor, AffineFakeQuantizedTensor):
        weight_tensor = weight_tensor.fq_data
    if bias is not None:
        return func(bias, input_tensor, weight_tensor)
    else:
        return func(input_tensor, weight_tensor)

@implements([aten.detach.default])
def _(func, types, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )

@implements([aten.clone.default])
def _(func, types, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )

@implements([aten._to_copy.default])
def _(func, types, *args, **kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )

@implements([aten.t.default])
def _(func, types, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.t)
    )

to_affine_fake_quantized = AffineFakeQuantizedTensor.from_float
