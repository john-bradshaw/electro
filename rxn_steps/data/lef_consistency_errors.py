



class NonLinearTopologyException(RuntimeError):
    """
    The atoms can be lined up end to end on the path
    """
    pass


class NotAddingAndRemovingError(RuntimeError):
    """
   The actions alternate between add and remove.
    """
    pass


class InconsistentActionError(RuntimeError):
    """
   The final molecule created by editing the reactants according to the action path
    """
    pass
