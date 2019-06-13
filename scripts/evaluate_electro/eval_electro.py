
import argparse

import torch
import numpy as np
import tqdm

from graph_neural_networks.core import utils

from rxn_steps.data import lef_uspto
from rxn_steps.model import get_electro
from rxn_steps.data import transforms as r_trsfms
from rxn_steps.predict import beam_searcher


class Params:
    def __init__(self):
        parser = argparse.ArgumentParser("Evaluate ELECTRO (or ELECTRO-LITE) on USPTO")
        parser.add_argument("checkpoint_path", help="location of the checkpoint file, use the string 'none' for random weights")
        parser.add_argument("output_file", help="where to store the predicted electron paths")
        parser.add_argument("--test_on_val", action="store_true", help="if set then will use validation dataset rather"
                                                                       "than the test dataset")
        parser.add_argument("--run_first_x", default=0, type=int, help="number of test set to use, (0 means run all)")
        args = parser.parse_args()

        self.chkpt_loc = args.checkpoint_path
        self.output_location = args.output_file
        self.use_val_as_test_set = args.test_on_val
        self.num_test_set_to_use = args.run_first_x

        self.beam_width = 10
        self.cuda_details = utils.CudaDetails(use_cuda=torch.cuda.is_available())


def _get_model_and_loadin_weights(cuda_details: utils.CudaDetails, params: Params):

    if params.chkpt_loc is not "none":
        print("Loading in weights from: {}".format(params.chkpt_loc))
        if not cuda_details.use_cuda:
            checkpoint = torch.load(params.chkpt_loc, map_location='cpu')
        else:
            checkpoint = torch.load(params.chkpt_loc)
    else:
        print("No weights found..")
        checkpoint = None

    if any(['reagents_net_aggregator' in k for k in checkpoint['state_dict'].keys()]):
        print("Using ELECTRO.")
        variant = get_electro.ElectroVariants.ELECTRO
    else:
        print("Using ELECTRO-LITE.")
        variant = get_electro.ElectroVariants.ELECTRO_LITE

    full_model = get_electro.get_model(variant, cuda_details=cuda_details)
    full_model = cuda_details.return_cudafied(full_model)

    if checkpoint is not None:
        full_model.load_state_dict(checkpoint['state_dict'])
    return full_model


def _get_data(use_val):
    tsfms = r_trsfms.TransformStrToBrokenDownParts()
    if use_val:
        print("Using USPTO LEF validation dataset")
        dataset = lef_uspto.LEFUspto(variant=lef_uspto.DataVariants.VAL, transform=tsfms)
    else:
        print("Using USPTO LEF test dataset")
        dataset = lef_uspto.LEFUspto(variant=lef_uspto.DataVariants.TEST, transform=tsfms)
    return dataset


def _convert_list__of_list_of_ints_to_txt(list_of_ints):
    list_of_strs = [','.join(map(str, elems)) for elems in list_of_ints]  # individual actions are separated by commas
    str_ = ';'.join(list_of_strs)  # top predictions are separated by semicolons.
    return str_


def main(params: Params):
    # We first load in the model
    electro = _get_model_and_loadin_weights(params.cuda_details, params)

    # Then the dataset
    dataset = _get_data(params.use_val_as_test_set)

    # Then we create the beam searcher
    predictor = beam_searcher.PredictiveRanking(electro, params.cuda_details)

    # Then we go through and predict out the series of electron paths for each reaction
    if params.num_test_set_to_use == 0:
        print("Running on whole dataset")
        num_points_to_eval = len(dataset)
    else:
        num_points_to_eval = params.num_test_set_to_use
        print(f"Running on the first {num_points_to_eval} of the dataset")
    MAX_TOP_ACC_TO_EVAL = 10
    acc_storage = np.zeros((num_points_to_eval, MAX_TOP_ACC_TO_EVAL))
    result_lines = []
    for i in tqdm.tqdm(range(num_points_to_eval)):
        data: r_trsfms.BrokenDownParts = dataset[i]

        # We do beam search out.
        predicted_result = predictor.predict_out(data.reactants, data.reagents)
        predicted_action_seqs = predictor.convert_predictions_to_ordered_am_path(predicted_result)

        # we store the results as a text string for further use later:
        result_lines.append(_convert_list__of_list_of_ints_to_txt(predicted_action_seqs))

        # we also work out whether any of the predicted paths match the ground truth one
        true_action_seq = data.ordered_am_path
        for j, pred_action_seq in enumerate(predicted_action_seqs[:MAX_TOP_ACC_TO_EVAL]):
            if pred_action_seq == true_action_seq:
                acc_storage[i, j] = 1.
                break

    # We now compute the path level average accuracies and print these out.
    top_k_accs = np.mean((np.cumsum(acc_storage, axis=1) > 0.5).astype(np.float64), axis=0)
    for k, k_acc in enumerate(top_k_accs, start=1):
        print(f"The top-{k} accuracy is {k_acc}")

    # Finally we store the reaction paths in a text file.
    with open(params.output_location, 'w') as fo:
        fo.writelines('\n'.join(result_lines))


if __name__ == '__main__':
    main(Params())
