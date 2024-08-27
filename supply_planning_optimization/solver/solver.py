import pulp
import numpy as np
import random
import warnings

from deap import base, creator, tools
from typing import Tuple
from abc import ABC, abstractmethod
from supply_planning_optimization.preprocessing.preprocessing import Preprocessing

warnings.filterwarnings("ignore")


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


class GeneticAlgorithmSolver(Solver, Preprocessing):
    def __init__(self) -> None:
        """
        Initializes the Genetic Algorithm solver with preprocessed data.
        """
        Preprocessing.__init__(self)
        self.inprice_data, self.outprice_data, self.demand_data = super().preprocess()
        self.inprice_data = self.inprice_data.iloc[:, 1:]
        self.outprice_data = self.outprice_data.iloc[:, 1:]
        self.demand_data = self.demand_data.iloc[:, 1:]

        self.num_plants = 2
        self.num_platforms = 2
        self.num_customers = 200
        self.toolbox = base.Toolbox()
        self.population = None
        self.best_inbound = None
        self.best_outbound = None

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
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        def init_individual():
            inbound = np.random.randint(
                0, 1000, size=(self.num_plants, self.num_platforms)
            )
            outbound = np.random.randint(
                0, 1000, size=(self.num_platforms, self.num_customers)
            )
            return creator.Individual(
                np.hstack((inbound.flatten(), outbound.flatten()))
            )

        self.toolbox.register("individual", init_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
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

        def evaluate(individual):
            individual = np.array(individual)
            inbound = individual[: self.num_plants * self.num_platforms].reshape(
                (self.num_plants, self.num_platforms)
            )
            outbound = individual[self.num_plants * self.num_platforms :].reshape(
                (self.num_platforms, self.num_customers)
            )

            total_cost = np.sum(inbound * self.inprice_data.values) + np.sum(
                outbound * self.outprice_data.values
            )

            penalty = 0

            for j in range(self.num_customers):
                if np.sum(outbound[:, j]) < self.demand_data.loc[j, "DEMAND"]:
                    penalty += 1e6

            for p in range(self.num_platforms):
                if np.sum(inbound[:, p]) != np.sum(outbound[p, :]):
                    penalty += 1e6

            return (total_cost + penalty,)

        self.toolbox.register("evaluate", evaluate)

    def implement_constraints(self) -> None:
        """
        The goal of this method is to implement
        constraints for the optimization process

        Arguments:
            -None
        Returns:
            -None
        """
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=10000, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def solve(self) -> None:
        """
        The goal of this method is to launch
        the solver optimization

        Arguments:
            -None
        Returns
            -None
        """
        self.population = self.toolbox.population(n=300)
        CXPB, MUTPB = 0.7, 0.8
        NGEN = 5000

        for gen in range(NGEN):
            offspring = self.toolbox.select(self.population, len(self.population))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            self.population[:] = offspring

            fits = [ind.fitness.values[0] for ind in self.population]

            if gen % 20 == 0:
                print(f"Generation {gen}: Min cost = {min(fits)}")

    def get_results(self) -> Tuple[np.array]:
        """
        The goal of this method is to get
        the optimization results once computed

        Arguments:
            -None
        Returns
            -inbound: np.array: The optimal inbound
            matrix
            -outbound: np.array: The optimal outbound
            matrix
        """
        best_ind = tools.selBest(self.population, 1)[0]

        self.best_inbound = np.array(
            best_ind[: self.num_plants * self.num_platforms]
        ).reshape((self.num_plants, self.num_platforms))
        self.best_outbound = np.array(
            best_ind[self.num_plants * self.num_platforms :]
        ).reshape((self.num_platforms, self.num_customers))

        return self.best_inbound, self.best_outbound


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
            -inbound: np.array: The optimal inbound
            matrix
            -outbound: np.array: The optimal outbound
            matrix
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
