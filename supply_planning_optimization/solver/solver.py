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
    def implement_constraints(self) -> None:
        """
        The goal of this method is to implement
        an abstract method for constraint implementation
        in the optimization process

        Arguments:
            -None
        Returns:
            -None
        """
        pass

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
