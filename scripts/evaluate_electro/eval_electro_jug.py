"""
A version of the evaluation script that uses Jug to run the calculations in parallel.
"""

import numpy as np
from jug import TaskGenerator


from graph_neural_networks.core import utils

from rxn_steps.data import lef_uspto
from rxn_steps.model import get_electro
from rxn_steps.data import transforms as r_trsfms
from rxn_steps.predict import beam_searcher

import eval_electro

class Params:
    def __init__(self):
        self.chkpt_loc = "../train_electro/chkpts/electro.pth.pick"
        self.beam_width = 10
        self.cuda_details = utils.CudaDetails(use_cuda=False)
        self.use_val_as_test_set = False
params = Params()

# We first load in the model
electro = eval_electro._get_model_and_loadin_weights(params.cuda_details, params)

# Then the dataset
dataset = eval_electro._get_data(params.use_val_as_test_set)

# Then we create the beam searcher
predictor = beam_searcher.PredictiveRanking(electro, params.cuda_details)

MAX_TOP_ACC_TO_EVAL = 10


# Using jug, create a function that works on one of the data points
@TaskGenerator
def _worker_func(data_index):
    data: r_trsfms.BrokenDownParts = dataset[data_index]

    # We do beam search out.
    predicted_result = predictor.predict_out(data.reactants, data.reagents)
    predicted_action_seqs = predictor.convert_predictions_to_ordered_am_path(predicted_result)

    # we store the results as a text string for further use later:
    result_str = eval_electro._convert_list__of_list_of_ints_to_txt(predicted_action_seqs)

    # we also work out whether any of the predicted paths match the ground truth one
    true_action_seq = data.ordered_am_path
    acc_storage = np.zeros(MAX_TOP_ACC_TO_EVAL)
    for j, pred_action_seq in enumerate(predicted_action_seqs[:MAX_TOP_ACC_TO_EVAL]):
        if pred_action_seq == true_action_seq:
            acc_storage[j] = 1.
            break
    return acc_storage, result_str

# Then we go through and predict out the series of electron paths for each reaction

num_to_use = len(dataset)
results = [_worker_func(i) for i in range(num_to_use)]