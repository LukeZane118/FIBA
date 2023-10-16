import argparse
import copy
import datetime
import logging
import os
import random
import test
import time

import numpy as np
import torch
import visdom
import yaml

import config
import train
import utils.csv_record as csv_record
from image_helper import ImageHelper
from utils.utils import dict_html, init_seed


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logger = logging.getLogger("logger")

vis = visdom.Visdom(port=8098)
criterion = torch.nn.CrossEntropyLoss()
init_seed(2023, False)

def trigger_test_byindex(helper, index, vis, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_index_" + str(index) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    if helper.params['vis_trigger_split_test']:
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=epoch_loss,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_index_" + str(index) + "_trigger")
def trigger_test_byname(helper, agent_name_key, vis, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_" + str(agent_name_key) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    if helper.params['vis_trigger_split_test']:
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=epoch_loss,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_" + str(agent_name_key) + "_trigger")
def vis_agg_weight(helper,names,weights,epoch,vis,adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0,len(names)):
        _name= names[i]
        _weight=weights[i]
        _is_poison=False
        if _name in adversarial_name_keys:
            _is_poison=True
        helper.target_model.weight_vis(vis=vis,epoch=epoch,weight=_weight, eid=helper.params['environment_name'],
                                       name=_name,is_poisoned=_is_poison)

def vis_fg_alpha(helper,names,alphas,epoch,vis,adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0,len(names)):
        _name= names[i]
        _alpha=alphas[i]
        _is_poison=False
        if _name in adversarial_name_keys:
            _is_poison=True
        helper.target_model.alpha_vis(vis=vis,epoch=epoch,alpha=_alpha, eid=helper.params['environment_name'],
                                       name=_name,is_poisoned=_is_poison)
        
def update_config_from_command(config, args: argparse.Namespace):
    for k, v in args._get_kwargs():
        if v is not None and k in config:
            config[k] = v
    return config

if __name__ == '__main__':
    print('Start training')
    np.random.seed(1)
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    parser.add_argument('-pf', '--postfix')
    parser.add_argument('-ts', '--target_ssim', type=float)
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)
    params_loaded = update_config_from_command(params_loaded, args)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded, name=params_loaded['type'])
    helper.load_data()
    helper.init_trigger()

    logger.info(f'load data done')
    helper.create_model()
    logger.info(f'create model done')
    adversary_list = helper.params['adversary_list']
    adversary_set = set(adversary_list)
    ### Create models
    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(adversary_list)}")

    best_loss = float('inf')

    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    submit_update_dict = None
    num_no_progress = 0
    adver_idx2poison_epochs = dict([(idx, set(helper.params[f'{idx}_poison_epochs'])) for idx in range(len(adversary_list))])
    
    example_images = helper.test_dataset[list(range(0, 91, 10))]

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()

        agent_name_keys = helper.participants_list
        adversarial_name_keys = []
        if helper.params['is_random_namelist']:
            if helper.params['is_random_adversary']:  # random choose, maybe don't have advasarial
                agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                adversarial_name_keys.extend(set(agent_name_keys) & adversary_set)
            else:  # must have advasarial if this epoch is in their poison epoch
                if helper.params['is_poison']:
                    ongoing_epochs = set(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                    for i, adversarial_name_key in enumerate(adversary_list):
                        if len(adver_idx2poison_epochs[i] & ongoing_epochs):
                            adversarial_name_keys.append(adversarial_name_key)

                nonattacker = list(adversary_set - set(adversarial_name_keys))
                benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(helper.benign_namelist + nonattacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params['is_random_adversary'] == False:
                adversarial_name_keys = copy.deepcopy(adversary_list)
        logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')
        epochs_submit_update_dict, num_samples_dict = train.train(helper=helper, start_epoch=epoch,
                                                                  local_model=helper.local_model,
                                                                  target_model=helper.target_model,
                                                                  is_poison=helper.params['is_poison'],
                                                                  agent_name_keys=agent_name_keys)
        logger.info(f'time spent on training: {time.time() - start_time}')
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict, adversarial_name_keys)
        is_updated = True
        if helper.params['aggregation_methods'] == config.AGGR_MEAN:
            # Average the models
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'])
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, updates, maxiter=maxiter)
            vis_agg_weight(helper, names, weights, epoch, vis, adversarial_name_keys)
            vis_fg_alpha(helper, names, alphas, epoch, vis, adversarial_name_keys)

        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)
            vis_agg_weight(helper,names,weights,epoch,vis,adversarial_name_keys)
            vis_fg_alpha(helper,names,alphas,epoch,vis,adversarial_name_keys )
            num_oracle_calls = 1

        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       visualize=True, agent_name_key="global")
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        if len(csv_record.scale_temp_one_row)>0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

        if helper.params['is_poison']:

            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                    epoch=temp_global_epoch,
                                                                                    model=helper.target_model,
                                                                                    is_poison=True,
                                                                                    visualize=False,
                                                                                    agent_name_key="global")

            csv_record.posiontest_result.append(
                ["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])


            # test on local triggers
            csv_record.poisontriggertest_result.append(
                ["global", "combined", "", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
            if helper.params['vis_trigger_split_test']:
                helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc_p, loss=epoch_loss,
                                                           eid=helper.params['environment_name'],
                                                           name="global_combined")
            if len(adversary_list) == 1:  # centralized attack
                if helper.params['attack_method'] == 'DBA' and helper.params['centralized_test_trigger'] == True:  # DBA centralized attack test on local triggers
                    for j in range(0, helper.params['trigger_num']):
                        trigger_test_byindex(helper, j, vis, epoch)
            else:  # distributed attack
                if helper.params['trigger_num'] > 1:
                    for agent_name_key in adversary_list:
                        trigger_test_byname(helper, agent_name_key, vis, epoch)
            
        helper.save_model(epoch=epoch)
        helper.save_trigger(epoch=epoch, example_images=example_images)
        logger.info(f'Done in {time.time() - start_time} sec.')
        csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)
        if helper.params['use_tensorboard']:
            helper.log_in_tensorboard(epoch)
        
        if epoch in helper.params['hessian_analysis_epoch']:
            helper.hessian_analysis(epoch)

    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")


    vis.save([helper.params['environment_name']])
