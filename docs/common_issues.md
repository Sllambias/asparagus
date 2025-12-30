
### torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
Example error: 
```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
JSONDecodeError: Extra data: line 1 column 182 (char 181)
Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
[W 2025-11-28 11:30:51,861] Trial 18 failed with value None.
Trial 18 failed: backend='inductor' raised:
JSONDecodeError: Extra data: line 1 column 182 (char 181)
Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
```

Solution:
```export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1```

### CUDA Device does not support bfloat16.
Example error:

Solution:
```hardware.precision=float32```

### NumExpr detected 48 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
See [CUDA Device does not support bfloat16](#cuda-device-does-not-support-bfloat16.)
