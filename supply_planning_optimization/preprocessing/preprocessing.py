import pandas as pd
from typing import Tuple


class Preprocessing:
    """
    The goal of this class is to preprocess
    files before they are fed to the solver

    Arguments:
        -None
    Returns:
        -None
    """

    def __init__(self) -> None:
        self.demand_data = pd.read_csv("df_demand.csv")
        self.inprice_data = pd.read_csv("df_inprice.csv")
        self.outprice_data = pd.read_csv("df_outprice.csv")

    def preprocess(self) -> Tuple[pd.DataFrame]:
        """
        The goal of this method is to preprocess
        the dataframes by removing the first useless
        columns for each

        Arguments:
            -None
        Returns:
            -None
        """

        self.inprice_data = self.inprice_data.iloc[:, 1:]
        self.outprice_data = self.outprice_data.iloc[:, 1:]
        self.demand_data = self.demand_data.iloc[:, 1:]

        return self.inprice_data, self.outprice_data, self.demand_data
