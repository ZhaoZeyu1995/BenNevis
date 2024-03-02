"""
Miscellaneous utility functions.

Functions
---------
dynamic_import
    Dynamically import a class from a module.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import importlib


def dynamic_import(module_name: str, class_name: str) -> type:
    """
    Dynamically import a class from a module.

    Arguments
    ---------
    module_name : str
        The name of the module to import from.
    class_name : str
        The name of the class to import.

    Returns
    -------
    class_: type
        The class that was imported.
    """
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_
