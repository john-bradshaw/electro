#!/usr/bin/env bash

# for running the jupyter notebook in Docker


PYTHONPATH=/rxn-steps/submodules/GNN:/rxn-steps/:${PYTHONPATH} /opt/conda/envs/py36/bin/jupyter notebook "$@"
