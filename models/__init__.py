#!/usr/bin/python3
"""models holds the models to run. The package pulls all the models into a dictionary of model."""

import os
from os import path

wd = path.dirname(path.realpath(__file__))
all_models = [x for x in os.listdir(wd) if ('.py' in x and '.pyc' not in x and '__' not in x)]
all_models = [path.splitext(x)[0] for x in all_models]
model = {}
for x in all_models:
    model[x] = getattr(__import__("models." + x , fromlist=x), "build_model")

