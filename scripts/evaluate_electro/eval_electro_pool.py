"""
A version of the evaluation script that works in parallel instead.
Run from this location so it can pick up the main eval_electro module.
"""

from multiprocessing import Pool

import numpy as np
import tqdm

from graph_neural_networks.core import utils

from rxn_steps.data import transforms as r_trsfms
from rxn_steps.predict import beam_searcher

import eval_electro


# Create a function that works on one of the data points
def _worker_func(arg):
    data_index, dataset, predictor, MAX_TOP_ACC_TO_EVAL = arg
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


def main():
    params = eval_electro.Params()
    params.cuda_details = utils.CudaDetails(use_cuda=False)  # will not use GPUs when work in paraellel
    params.num_workers = 17

    # We first load in the model
    electro = eval_electro._get_model_and_loadin_weights(params.cuda_details, params)

    # Then the dataset
    dataset = eval_electro._get_data(params.use_val_as_test_set)

    # Then we create the beam searcher
    predictor = beam_searcher.PredictiveRanking(electro, params.cuda_details)

    # Then we go through and predict out the series of electron paths for each reaction
    assert params.num_test_set_to_use == 0, "should be run on whole dataset"
    MAX_TOP_ACC_TO_EVAL = 10

    num_to_use = len(dataset)
    # Create a pool and assign the workers to go through the dataset
    pool = Pool(params.num_workers)
    list_of_results = list(tqdm.tqdm(pool.imap(_worker_func, ((i, dataset, predictor, MAX_TOP_ACC_TO_EVAL) for i in range(num_to_use))),
                                     total=num_to_use))
    pool.close()
    pool.join()

    # Stitch the results back together:
    top_k_accs, result_lines = zip(*list_of_results)
    acc_storage = np.stack(top_k_accs)

    # We now compute the path level average accuracies and print these out.
    top_k_accs = np.mean((np.cumsum(acc_storage, axis=1) > 0.5).astype(np.float64), axis=0)
    for k, k_acc in enumerate(top_k_accs, start=1):
        print(f"The top-{k} accuracy is {k_acc}")

    # Finally we store the reaction paths in a text file.
    with open(params.output_location, 'w') as fo:
        fo.writelines('\n'.join(result_lines))


if __name__ == '__main__':
    main()
