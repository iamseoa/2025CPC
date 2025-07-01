import numpy as np

def count_parameters(params):
    """
    params: list of cupy arrays (model parameters)
    반환: (파라미터 총 개수, 파라미터 크기(MB))
    """
    total_params = 0
    total_bytes = 0
    for p in params:
        total_params += p.size
        total_bytes += p.nbytes
    size_mb = total_bytes / (1024 ** 2)
    return total_params, size_mb

def print_model_size(params, model_name="Model"):
    total_params, size_mb = count_parameters(params)
    print(f"{model_name} - Total parameters: {total_params:,}")
    print(f"{model_name} - Model size: {size_mb:.2f} MB")

