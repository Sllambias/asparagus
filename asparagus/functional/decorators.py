from functools import wraps
from importlib.util import find_spec


def depends_on(package):
    """Raise an ImportError when the decorated function is called without the optional package installed."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            spec = find_spec(package)
            if spec is None:
                raise ImportError(f"Optional dependency {package} not found ({func.__name__}).")
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
