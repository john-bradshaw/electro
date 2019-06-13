#!/usr/bin/env bash
# You should source this script to get the correct additions to Python path


# Directory of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set up the Python Paths
export PYTHONPATH=${PYTHONPATH}:${DIR}/submodules/GNN

export PYTHONPATH=${PYTHONPATH}:${DIR}