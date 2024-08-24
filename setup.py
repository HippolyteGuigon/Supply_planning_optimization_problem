from setuptools import setup, find_packages

setup(
    name="supply_planning_optimization",
    version="0.1.0",
    packages=find_packages(
        include=["supply_planning_optimization", "supply_planning_optimization.*"]
    ),
    description="Python programm for creating an optimization\
        algorithm for supply chain planning",
    author="Hippolyte Guigon",
    author_email="Hippolyte.guigon@hec.edu",
    url="https://github.com/HippolyteGuigon/Supply_planning_optimization_problem",
)
