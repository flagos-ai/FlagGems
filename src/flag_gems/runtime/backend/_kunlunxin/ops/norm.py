from .vector_norm import vector_norm


def norm(x, p=2, dim=None, keepdim=False):
    return vector_norm(x, ord=2 if p is None else p, dim=dim, keepdim=keepdim)


def norm_scalar(x, p=2):
    return norm(x, p=p, dim=None, keepdim=False)


def norm_scalaropt_dim(x, p, dim, keepdim=False):
    return norm(x, p=p, dim=dim, keepdim=keepdim)
