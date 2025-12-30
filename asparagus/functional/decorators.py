from functools import wraps
from importlib.util import find_spec


def depends_on_mlflow():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            spec = find_spec("mlflow")
            if spec is None:
                raise ImportError(f"Optional dependency mlflow not found ({func.__name__}).")
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def depends_on_timm():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            spec = find_spec("timm")
            if spec is None:
                raise ImportError(f"Optional dependency timm not found ({func.__name__}).")
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
