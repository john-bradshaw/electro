#!/usr/bin/env bash

# for running the jupyter notebook in Docker


PYTHONPATH=/electro/submodules/GNN:/electro/:${PYTHONPATH} /opt/conda/envs/py36/bin/jupyter notebook "$@"
