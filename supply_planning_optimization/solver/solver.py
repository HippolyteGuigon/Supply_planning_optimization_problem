from abc import ABC, abstractmethod


class Solver(ABC):
    """
    The goal of this class is to implement
    an abstract class for all solver techniques
    that will then be used

    Arguments:
        -None
    Returns:
        -None
    """

    @abstractmethod
    def solve(self) -> None:
        """
        The goal of this method is to have an abstract
        method to launch the solver optimization

        Arguments:
            -None
        Returns
            -None
        """
        pass
