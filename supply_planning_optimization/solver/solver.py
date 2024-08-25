import pulp
import numpy as np

from typing import Tuple
from abc import ABC, abstractmethod
from supply_planning_optimization.preprocessing.preprocessing import Preprocessing


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
    def define_objective_function(self) -> None:
        """
        The goal of this method is to implement an
        abstract method for defining the objective function
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

    @abstractmethod
    def get_results(self) -> None:
        """
        The goal of this method is to have an abstract
        method to get the optimization results

        Arguments:
            -None
        Returns
            -None
        """
        pass


class SimplexSolver(Solver, Preprocessing):
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
        Initializes the Simplex solver with a specific
        linear programming problem.

        Arguments:
            -None
        Returns:
            -None
        """

        Preprocessing.__init__(self)

        self.inprice_data, self.outprice_data, self.demand_data = super().preprocess()
        self.solver = pulp.LpProblem("Transhipment_Problem", pulp.LpMinimize)

    def decision_variable_definition(self) -> None:
        """
        The goal of this method is to define the
        decision variables for the optimization
        process

        Arguments:
            -None
        Returns:
            -None
        """

        self.Inbound = pulp.LpVariable.dicts(
            "Inbound",
            [(i + 1, j + 1) for i in range(2) for j in range(2)],
            lowBound=0,
            upBound=None,
            cat="Integer",
        )

        self.Outbound = pulp.LpVariable.dicts(
            "Outbound",
            [(i + 1, j + 1) for i in range(2) for j in range(200)],
            lowBound=0,
            upBound=None,
            cat="Integer",
        )

    def define_objective_function(self) -> None:
        """
        The goal of this method is to  define
        the objective function in the optimization
        process

        Arguments:
            -None
        Returns:
            -None
        """

        self.solver += pulp.lpSum(
            [
                self.inprice_data.iloc[i, j + 1] * self.Inbound[i + 1, j + 1]
                for i in range(2)
                for j in range(2)
            ]
        ) + pulp.lpSum(
            [
                self.outprice_data.iloc[i, j + 1] * self.Outbound[i + 1, j + 1]
                for i in range(2)
                for j in range(200)
            ]
        )

    def implement_constraints(self) -> None:
        """
        The goal of this method is to implement
        constraints for the optimization process

        Arguments:
            -None
        Returns:
            -None
        """

        for j in range(200):
            self.solver += (
                pulp.lpSum([self.Outbound[i + 1, j + 1] for i in range(2)])
                >= self.demand_data.loc[j, "DEMAND"]
            )
        for p in range(2):
            self.solver += pulp.lpSum(
                [self.Inbound[i + 1, p + 1] for i in range(2)]
            ) == pulp.lpSum([self.Outbound[p + 1, j + 1] for j in range(200)])

    def solve(self) -> None:
        """
        The goal of this method is to launch
        the solver optimization

        Arguments:
            -None
        Returns
            -None
        """

        self.status = self.solver.solve()

    def get_results(self) -> Tuple[np.array]:
        """
        The goal of this method is to get
        the optimization results once computed

        Arguments:
            -None
        Returns
            -None
        """

        inbound = np.array(
            [[self.Inbound[i + 1, j + 1].varValue for j in range(2)] for i in range(2)]
        )
        outbound = np.array(
            [
                [self.Outbound[i + 1, j + 1].varValue for j in range(200)]
                for i in range(2)
            ]
        )

        return inbound, outbound
