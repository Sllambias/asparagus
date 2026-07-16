from hydra.utils import instantiate


def fast_instantiate(cfg, *args, **kwargs):
    """
    hydra instantiate deep copies all config objects which can be extremely slow.
    Instead we use a partial instantiation approach to avoid deep copying the config objects.
    """
    instance = instantiate(cfg, _partial_=True)
    instance = instance(*args, **kwargs)
    return instance
