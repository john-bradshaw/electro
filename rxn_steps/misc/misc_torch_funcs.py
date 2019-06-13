

def isin(ar1, ar2):
    """
    same as numpy.isin
    see: https://github.com/pytorch/pytorch/issues/3025
    """
    return (ar1[..., None] == ar2).any(-1)



