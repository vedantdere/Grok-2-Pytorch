import torch
from torch import nn

class LinearBase(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 skip_bias_add=False,
                 params_dtype=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype or torch.get_default_dtype()

    def forward(self,x):
        raise NotImplementedError
    
class ReplicatedLinear(LinearBase):
    def __init__(
            self,
            input_size,
            output_size,
            bias=True,
            skip_bias_add=False,
            params_dtype=None
    ):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype
        )

        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=self.params_dtype))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(output_size, dtype=self.params_dtype)
            )
        else:
            self.bias = None

    def forward(self,
                x):
        output = x @ self.weight.t()

        if not self.skip_bias_add and self.bias is not None:
            out = out + self.bias
            return out , None
        else:
            return output, self.bias
        
class ColumnParallelLinear(LinearBase):
    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 gather_output=False,
                 skip_bias_add=False,
                 params_dtype=None,
                 tp_size=1,
                 tp_rank=1):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype
        )

        self.tp_size = tp_size
        self.tp_rank = tp_rank

        assert output_size % tp_size == 0

        self.output_size_per_partition = output_size // tp_size

        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition, input_size,
                dtype=self.params_dtype
            )
        )
        
        nn.init.kaiming_uniform_(self.weight,a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_size_per_partition, dtype=self.params_dtype))
        else:
            self.bias = None

        self.gather_output = gather_output

    def forward(self, x):
        output_parallel = x @ self.weight.t()
        if not self.skip_bias_add and self.bias is not None:
            output_parallel = output_parallel + self.bias
        
        if self.gather_output and self.tp_size > 1:
            outputs = [output_parallel for _ in range(self.tp_size)]
            out = torch.cat(outputs, dim=-1)
        else:
            out = output_parallel
        
        output_bias = self.bias if self.skip_bias_add else None
        return out, output_bias
    

class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self,
                 input_size,
                 output_sizes,
                 bias=True,
                 gather_output=False,
                 skip_bias_add=False,
                 params_dtype=None,
                 tp_size=1,
                 tp_rank=0):
        
        self.output_size = output_sizes
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            tp_size=tp_size,
            tp_rank=tp_rank
        )

    def forward(self,x):
        out , bias = super.forward(x)
        splits = torch.split(out, [s // self.tp_size for s in self.output_sizes],dim=-1)
        return list(splits), bias

class RowParallelLinear(LinearBase):
    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 input_is_parallel=True,
                 skip_bias_add=False,
                 params_dtype=None,
                 reduce_results=True,
                 tp_rank=0,
                 tp_size=1
                 ):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype
        )

        assert input_size % tp_size == 0

        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.input_size_per_partition = input_size // tp_size

        self.weight = nn.Parameter(
            torch.empty(
                self.input_size_per_partition, output_size, dtype=self.params_dtype
            )
        )

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(output_size, dtype=self.params_dtype)
            )
        else:
            self.bias = None

    def forward(self,x):
        if self.input_is_parallel:
            x_parallel = x

            assert x_parallel.size(-1) == self.input_size_per_partition
        else:
            chunks = torch.chunk(x, self.tp_size,dim=-1)
            x_parallel = chunks[self.tp_rank].contiguous()

        out_parallel = x_parallel @ self.weight

        if self.bias is not None and not self.skip_bias_add and self.tp_rank == 0:
            out_parallel = out_parallel + self.bias

        out = out_parallel
        out_bias = self.bias if self.skip_bias_add else None
        return out , out_bias
    
    