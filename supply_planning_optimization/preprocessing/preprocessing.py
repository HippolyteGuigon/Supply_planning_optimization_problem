import pandas as pd


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
