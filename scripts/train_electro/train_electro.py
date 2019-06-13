

import argparse
import time
import collections

import tqdm
import torch
import numpy as np
from torch.nn import functional as F
from torch import optim
from torch.utils import data
from torchvision import transforms as t_trsfms
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.WARNING)

from graph_neural_networks.core import utils

from rxn_steps.data import lef_uspto
from rxn_steps.data import transforms as r_trsfms

from rxn_steps.model import get_electro

STR_SEP = "=" * 20


class Params:
    def __init__(self):
        parser = argparse.ArgumentParser("Train ELECTRO (or ELECTRO-LITE) on USPTO-LEF dataset")
        parser.add_argument("--electro_lite", action="store_true")
        args = parser.parse_args()

        self.electro_lite_flag = args.electro_lite
        self.num_epochs = 14
        self.initial_lr = 0.001
        self.lr_decay_epochs = [8, 12]
        self.lr_decay_factor = 0.1
        self.batch_size_wrt_reactions = 30
        self.val_batch_size_wrt_reactions = 100

        self.cuda_details = utils.CudaDetails(use_cuda=torch.cuda.is_available())
        self.num_dataloader_workers = 10



def electro_forward_to_loss(full_model: get_electro.FullModel, data: r_trsfms.ElectroTrainInput):
    # Embed the nodes in the relevant graphs
    graphs_w_nodes_embedded = full_model.ggnn(data.graphs)
    reagents_w_nodes_embedded = None if data.reagents is None else full_model.ggnn(data.reagents)

    # Compute the logits for the respective actions
    stop_logits, initial_action_logits, remove_action_logits, add_action_logits = full_model.electro(
        graphs=graphs_w_nodes_embedded,
        initial_select_inputs=data.initial_input,
        remove_select_inputs=data.remove_input,
        add_select_inputs=data.add_input, reagent_graphs=reagents_w_nodes_embedded
    )

    # Now can do the losses
    losses_ = {}

    # Compute the loss due to stop
    assert len(stop_logits.shape) == 1 and len(data.stop_label.shape) == 1
    loss_stop = F.binary_cross_entropy_with_logits(input=stop_logits, target=data.stop_label.float(), reduction='none')
    loss_stop[~data.stop_mask] = 0.
    loss_stop_total = loss_stop.sum() / data.num_reactions
    losses_['stop'] = loss_stop_total

    # Compute the loss due to initial, add, remove
    results = []
    for name, logits, targets in [("initial", initial_action_logits, data.initial_target),
                                  ("remove", remove_action_logits, data.remove_target),
                                  ("add", add_action_logits, data.add_target)
                                  ]:
        l_ = logits.nll_per_graph(targets.label_per_graph, targets.action_mask)
        l_ = l_.sum() / data.num_reactions
        losses_[name] = l_
        results.append(l_)
    loss_initial, loss_remove, loss_add = results

    # Add the losses together
    total_loss = loss_stop_total + loss_initial + loss_remove + loss_add
    return total_loss, losses_


def train(train_dataloader, full_model: get_electro.FullModel, cuda_details, optimizer):
    print("\n\033Train\033[0m")  # escape chars for underlining

    # set up meters record running averages
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # put in train mode
    full_model.train()

    # iterate through the train dataset
    end = time.time()
    print('\n')
    with tqdm.tqdm(train_dataloader, total=len(train_dataloader)) as t:
        for data in t:
            data_time.update(time.time() - end)

            # get the inputs onto correct device
            data: r_trsfms.ElectroTrainInput = data.to_torch(cuda_details)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            loss, _ = electro_forward_to_loss(full_model, data)

            # backward + optimize
            loss.backward()
            losses.update(loss.item(), data.num_reactions)
            optimizer.step()

            # print statistics
            batch_time.update(time.time() - end)
            end = time.time()

            # Update the stats in the progress bar
            t.set_postfix(loss=f"{losses.val:.4E}", loss_avg=f"{losses.avg:.4E}",
                          data_time=f"{data_time.avg:.3f}",
                batch_time=f"{batch_time.avg:.3f}")


def validate(dataloader_val, full_model: get_electro.FullModel, cuda_details):
    print("\n\033[4mValidation\033[0m")  # escape chars for underlining

    # set up meters record running averages
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    individual_losses = collections.defaultdict(lambda : utils.AverageMeter())

    # switch to evaluate mode
    full_model.eval()

    # iterate through the validation dataset
    print('\n')
    end = time.time()
    with tqdm.tqdm(dataloader_val, total=len(dataloader_val)) as t:
        for data in t:
            with torch.no_grad():
                # get the inputs onto correct device
                data: r_trsfms.ElectroTrainInput = data.to_torch(cuda_details)

                # get the loss
                loss, indi_losses_for_batch = electro_forward_to_loss(full_model, data)
                losses.update(loss.item(), data.num_reactions)

            for k, v in indi_losses_for_batch.items():
                individual_losses[k].update(v.item(), data.num_reactions)

            # update time stats
            batch_time.update(time.time() - end)
            end = time.time()

            # Update the stats in the progress bar
            t.set_postfix(loss=f"{losses.val:.4E}", loss_avg=f"{losses.avg:.4E}",
                batch_time=f"{batch_time.avg:.3f}")

    # Print the final averages
    print("\nTest: [end]\t"
              "Time Total: {batch_time.sum:.3f}\t"
              "Average Loss: {losses.avg:.4f}".format(
            batch_time=batch_time, losses=losses
        ))
    print(f"Individual losses: {str([(k, v.avg) for k,v in sorted(individual_losses.items())])}")
    return losses.avg


def main(params: Params):
    # Print out the parameters we will use.
    print(STR_SEP)
    print("Running ELECTRO train script.")
    variant_str = "ELECTRO" if not params.electro_lite_flag else "ELECTRO-LITE"
    print(f"Variant is {variant_str}. Params are: ")
    print(str(params.__dict__))
    print(STR_SEP)

    # Create the dataset/ dataloaders!
    tsfms = t_trsfms.Compose([
        r_trsfms.TransformStrToBrokenDownParts(),
        r_trsfms.TransformToRdKitIntermediates(),
        r_trsfms.TransformRdKitIntermediatesToElectroTrainInput(exclude_reagents_from_initial_flag=params.electro_lite_flag)
    ])
    train_dataset = lef_uspto.LEFUspto(variant=lef_uspto.DataVariants.TRAIN, transform=tsfms)
    val_dataset = lef_uspto.LEFUspto(variant=lef_uspto.DataVariants.VAL, transform=tsfms)
    train_dataloader = data.DataLoader(train_dataset, batch_size=params.batch_size_wrt_reactions, shuffle=False, #shuffle=True,
                                       num_workers=params.num_dataloader_workers, collate_fn=r_trsfms.ElectroBatchCollate())
    val_dataloader = data.DataLoader(val_dataset, batch_size=params.val_batch_size_wrt_reactions, shuffle=False,
                                       num_workers=params.num_dataloader_workers, collate_fn=r_trsfms.ElectroBatchCollate())

    # Create the model
    variant = get_electro.ElectroVariants.ELECTRO if not params.electro_lite_flag else get_electro.ElectroVariants.ELECTRO_LITE
    full_model = get_electro.get_model(variant, cuda_details=params.cuda_details)
    full_model = params.cuda_details.return_cudafied(full_model)

    # Create the optimizer
    optimizer = optim.Adam(full_model.parameters(), lr=params.initial_lr)
    lr_annealer = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.lr_decay_epochs, gamma=params.lr_decay_factor)

    # Train (and validate before every new epoch)!
    print(STR_SEP)
    print("Starting training...")
    for epoch in range(params.num_epochs):
        print(f"\n\n\n{STR_SEP} Starting epoch {epoch}.")
        validate(val_dataloader, full_model, params.cuda_details)
        lr_annealer.step()
        train(train_dataloader, full_model, params.cuda_details, optimizer)
        utils.save_checkpoint(dict(
            epoch=epoch+1,
            state_dict=full_model.state_dict(),
            optimizer=optimizer.state_dict()
        ), is_best=True)
        print(STR_SEP)

    print("Final validation:")
    validate(val_dataloader, full_model, params.cuda_details)
    print("Done!")


if __name__ == '__main__':
    main(Params())
