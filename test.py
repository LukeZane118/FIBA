import torch
import torch.nn as nn

import main


def Mytest(helper, epoch, model, is_poison=False, visualize=True, agent_name_key=""):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        data_iterator = helper.test_data
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(batch, evaluation=True)
            dataset_size += len(data)
            output = model(data)
            total_loss += criterion(output, targets).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / (float(dataset_size) if dataset_size!=0 else 0)

    main.logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, dataset_size,
                                                        acc))
    if visualize: # loss =total_l
        model.test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None,
                       eid=helper.params['environment_name'],
                       agent_name_key=str(agent_name_key))
    model.train()
    return (total_l, acc, correct, dataset_size)


def Mytest_poison(helper, epoch, model, is_poison=False, visualize=True, agent_name_key=""):
    """Test the attack success rate of combined trigger
    """
    model.eval()
    helper.set_trigger_to_eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        data_iterator = helper.test_data_poison
        # ssim = 0
        helper.imageQuality.reset()
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)
            # if is_frequency_attack:
                # ssim += helper.ssim_cal(batch[0].to(data.device), data)
            helper.imageQuality.compute(batch[0].to(data.device), data)
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            # total_loss += nn.functional.cross_entropy(output, targets,
            #                                         reduction='sum').item()  # sum up batch loss
            total_loss += criterion(output, targets).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / (float(dataset_size) if dataset_size!=0 else 0)
    main.logger.info('___Test {}&Combined trigger poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%), '.format(model.name, is_poison, epoch,
                                                        total_l, correct, poison_data_count,
                                                        acc) + \
                     ", ".join(["{}: {:.4f}".format(*metric_val) for metric_val in helper.imageQuality.get_average().items()])
                     )
    if visualize: #loss = total_l
        model.poison_test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None, eid=helper.params['environment_name'], agent_name_key=str(agent_name_key))

    model.train()
    helper.set_trigger_to_train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_trigger(helper, model, adver_trigger_index):
    """Test the attack success rate of trigger got by index
    """
    model.eval()
    helper.set_trigger_to_eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        data_iterator = helper.test_data_poison
        adv_index = adver_trigger_index
        
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            # total_loss += nn.functional.cross_entropy(output, targets,
            #                                         reduction='sum').item()  # sum up batch loss
            total_loss += criterion(output, targets).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / float(poison_data_count) if poison_data_count!=0 else 0

    model.train()
    helper.set_trigger_to_train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_agent_trigger(helper, model, agent_name_key):
    """Test the attack success rate of trigger got by name
    """
    model.eval()
    helper.set_trigger_to_eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    adversary_list = helper.params['adversary_list']
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        data_iterator = helper.test_data_poison
        # try:
        #     adv_index = adversary_list.index(agent_name_key)
        # except:
        #     adv_index = -1
        adv_index = adversary_list.index(agent_name_key)
        # for temp_index in range(0, len(adversary_list)):
        #     if int(agent_name_key) == adversary_list[temp_index]:
        #         adv_index = temp_index
        #         break
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            # total_loss += nn.functional.cross_entropy(output, targets,
            #                                         reduction='sum').item()  # sum up batch loss
            total_loss += criterion(output, targets).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / float(poison_data_count) if poison_data_count!=0 else 0

    model.train()
    helper.set_trigger_to_train()
    return total_l, acc, correct, poison_data_count
