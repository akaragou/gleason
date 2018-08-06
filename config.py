#!/usr/bin/env python3
"""Project config file. Example configurations can be found in the configs/
directory."""

import numpy as np
import os
import toml

class Config():
    """Config loads the configuration for a given run from a toml file."""

    def __init__(self, filename):
        """initialize the configuration with something reasonable if none is
        provided."""
        if filename is None:
            self.config = toml.load('configs/default.conf')
            return
        self.config = toml.load(filename)
        self.config['directory'] = {}
        self.config['directory']['root'] = os.path.dirname(
                os.path.realpath(__file__))
        self.config['directory']['datasets'] = os.path.join(
                self.config['directory']['root'], 'datasets')

    def _verify(self):
        """_verify will throw an exception if the constructed configuration is
        invalid."""
        pass

    def __getitem__(self, name):
        """gets an item from the read configuration."""
        return self.config[name]
