import pulp
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
    def decision_variable_definition(self) -> None:
        """
        The goal of this method is to implement an
        abstract method for defining decision variables
        in the optimization process

        Arguments:
            -None
        Returns:
            -None
        """
        pass

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


class SimplexSolver(Solver):
    """
    The goal of this class is to implement
    a Simplex solver method

    Arguments:
        -None
    Returns:
        -None
    """

    def __init__(self) -> None:
        """
        Initializes the Simplex solver with a specific linear programming problem.

        Arguments:
            -None
        Returns:
            -None
        """
        self.solver = pulp.LpProblem("Transhipment_Problem", pulp.LpMinimize)
