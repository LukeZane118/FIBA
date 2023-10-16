import copy
import datetime
import logging
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import yaml
from pyhessian import hessian  # Hessian computation
from torchvision import datasets, transforms

import config
from attacks import *
from config import device
from helper import Helper
from models.resnet_light import ResNet18
from models.resnet_tinyimagenet import resnet18
from utils.utils import ImageQuality, combine_dataset, fix_bn


logger = logging.getLogger("logger")
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ImageHelper(Helper):

    def create_model(self):
        local_model=None
        target_model=None
        if self.params['type'] == config.TYPE_CIFAR10:
            local_model = ResNet18(num_classes=self.params['num_classes'],name='Local',
                                   created_time=self.params['current_time'])
            target_model = ResNet18(num_classes=self.params['num_classes'],name='Target',
                                   created_time=self.params['current_time'])

        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            local_model= resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])

        local_model=local_model.to(device)
        target_model=target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available() :
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}", map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model.type(torch.float)    # add type casting to fixed bug
        self.target_model = target_model.type(torch.float)

    def build_classes_dict(self):
        classes = defaultdict(list)
        for ind, (_, label) in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            classes[label.item()].append(ind)
        return classes

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = self.classes_dict
        class_size = len(cifar_classes[0]) # for cifar: 5000
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())  # for cifar: 10

        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)
        # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
        return per_participant_list

    def draw_dirichlet_plot(self,no_classes,no_participants,image_nums,alpha):
        fig= plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()
        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))
        vis_par=[0,10,20,30]
        for k in range(no_participants):
        # for k in vis_par:
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            widths = s[k]
            xcenters = left + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # for y, (x, c) in enumerate(zip(xcenters, widths)):
            #     plt.text(x, y, str(int(c)), ha='center', va='center',
            #              color=text_color,fontsize='small')
            left += s[k]
        plt.legend(ncol=20,loc='lower left',  bbox_to_anchor=(0, 1),fontsize=4) #
        # plt.legend(ncol=len(vis_par), bbox_to_anchor=(0, 1),
        #            loc='lower left', fontsize='small')
        plt.xlabel("Number of Images", fontsize=16)
        # plt.ylabel("Label 0 ~ 199", fontsize=16)
        # plt.yticks([])
        fig.tight_layout(pad=0.1)
        # plt.ylabel("Label",fontsize='small')
        fig.savefig(self.folder_path+'/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = defaultdict(list)
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            test_classes[label].append(ind)

        # range_no_id = list(range(0, len(self.test_dataset)))
        # for image_ind in test_classes[self.params['poison_label_swap']]:
        #     if image_ind in range_no_id:
        #         range_no_id.remove(image_ind)
        range_no_id = list(set(range(len(self.test_dataset))) - set(test_classes[self.params['poison_label_swap']]))
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range_no_id)), \
               torch.utils.data.DataLoader(self.test_dataset,
                                            batch_size=self.params['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                poison_label_inds))
               
    def load_data(self):
        logger.info('Loading data')
        
        if self.params['type'] == config.TYPE_CIFAR10:
            
            dataPath = "../data/cifar10" # you can change to your data path
            
            trainset = datasets.CIFAR10(root=dataPath, train=True, download=True)
            testset = datasets.CIFAR10(root=dataPath, train=False, download=True)
            x_train, y_train = trainset.data, trainset.targets
            x_test, y_test = testset.data, testset.targets
            x_train = x_train.astype(np.float) / 255.
            x_test = x_test.astype(np.float) / 255.
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            x_train, x_test = np.transpose(x_train, (0, 3, 1, 2)), np.transpose(x_test, (0, 3, 1, 2))
            x_train, y_train = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long).view((-1, ))
            x_test, y_test = torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long).view((-1, ))

            self.train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
            self.test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
            
        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            _data_dir = '../data/tiny-imagenet-200/' # you can change to your data path
            trainset = datasets.ImageFolder(os.path.join(_data_dir, 'train'), transforms.ToTensor())
            testset = datasets.ImageFolder(os.path.join(_data_dir, 'val'), transforms.ToTensor())

            x_train, y_train = zip(*[sample for sample in trainset])
            x_test, y_test = zip(*[sample for sample in testset])
            x_train = torch.stack(x_train)
            x_test = torch.stack(x_test)
            y_train = torch.LongTensor(y_train).view((-1, ))
            y_test = torch.LongTensor(y_test).view((-1, ))
            self.train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
            self.test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
            
        logger.info('reading data done')

        self.classes_dict = self.build_classes_dict()
        logger.info('build_classes_dict done')
        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'], #100
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        else:
            ## sample indices for participants that are equally
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        logger.info('train loaders done')
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.test_data_poison, self.test_targetlabel_data = self.poison_test_dataset()

        self.advasarial_namelist = self.params['adversary_list']

        if self.params['is_random_namelist'] == False:
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['number_of_total_participants']))
        
        self.benign_namelist = list(set(self.participants_list) - set(self.advasarial_namelist))
        
    def init_trigger(self):
        if self.params['attack_method'] == 'DBA':
            self.triggers = [DBA(patterns=self.params[f"{i}_poison_pattern"], image_shape=self.params['image_shape']).to(device) for i in range(self.params['trigger_num'])]
        elif self.params['attack_method'] == 'FIBA':
            self.triggers = [FIBA(init_magnitude=self.params['init_magnitude'], pattern_shape=self.params['image_shape']).to(device) for _ in range(self.params['trigger_num'])]
        
        if len(self.triggers) > 1:  # combined trigger
            if self.params['attack_method'] == 'DBA':
                self.triggers.append(DBA().to(device))
                self.triggers[-1].set_by_combination_(self.triggers[:-1])

        self.first_attack = [True] * self.params['trigger_num']
        self.imageQuality = ImageQuality(self.params["image_metrics"])

        logger.info("Initializing trigger done.")
    
    def local_train_trigger(self, adversarial_index):
        if len(self.triggers) == 1:
            # set to 0 for getting first attacker's data when number of attacker more than 1
            adversarial_index = 0
        if not self.first_attack[adversarial_index]:
            return
        # self.construct_trigger(adversarial_index)
        self.first_attack[adversarial_index] = False

        if self.params['attack_method'] == 'DBA':
            return
        
        if self.params['resumed_trigger_name'] is not None:
            if torch.cuda.is_available():
                loaded_params = torch.load(f"saved_models/{self.params['resumed_trigger_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_trigger_name']}", map_location='cpu')
            self.triggers[adversarial_index].load_state_dict(loaded_params)
            logger.info("Loaded trained trigger.")
            return
            
        if self.params['use_all_attackers_data']:
            data_iterator = combine_dataset([self.train_data[agent_name_key][1] for agent_name_key in self.params['adversary_list']])
        else:
            _, data_iterator = self.train_data[self.params['adversary_list'][adversarial_index]]

        logger.info(f"Start training trigger {adversarial_index}...")
        if self.params['attack_method'] == 'FIBA':
            train_trigger(
                trigger = self.triggers[adversarial_index], 
                model = self.local_model,
                y_target = self.params["poison_label_swap"],
                device = device,
                dataloader = data_iterator,
                image_shape = self.params["image_shape"],
                lr = self.params["trigger_lr"],
                use_mean_att = self.params["use_mean_att"],
                att_fig_save_path = os.path.join(self.folder_path, f'wmse_att{adversarial_index}.png'),
                reg_max = self.params["reg_max"],
                reg_wmse = self.params["reg_wmse"],
                target_QoE = self.params["target_QoE"],
                n_iter = self.params["n_iter"],
                threshold = self.params["threshold"],
                target_layers = self.params["target_layers"],
                verbose = self.params["verbose"],
                )
        logger.info(f"Training trigger {adversarial_index} done.")

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices), pin_memory=True, num_workers=8)
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=False)
        return test_loader

    def get_batch(self, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def get_poison_batch(self, bptt, adversarial_index=-1, evaluation=False):
        
        if len(self.triggers) == 1:
            adversarial_index = -1

        images, targets = bptt
        images, targets = images.to(device), targets.to(device)

        poison_count = 0
        
        if evaluation: # poison all data when testing
            new_targets = torch.full_like(targets, self.params['poison_label_swap'])
            new_images = self.add_pattern(images, adversarial_index)
            poison_count = len(new_targets)

        else: # poison part of data when training
            poison_count = self.params['poisoning_per_batch']
            new_targets = targets.clone()
            new_targets[:poison_count] = self.params['poison_label_swap']
            new_images = images.clone()
            new_images[:poison_count] = self.add_pattern(images[:poison_count], adversarial_index)
            
        return new_images, new_targets, poison_count
        
    def add_pattern(self, ori_images, adversarial_index):
        images = copy.deepcopy(ori_images)
        with torch.no_grad():
            images = self.triggers[adversarial_index](images)
        return images
    
    def set_trigger_to_train(self):
        for trigger in self.triggers:
            trigger.train()
                
    def set_trigger_to_eval(self):
        if self.params['attack_method'] != 'DBA' and len(self.triggers) > 1:
            self.triggers[-1].set_by_combination_([trigger for i, trigger in enumerate(self.triggers[:-1]) if not self.first_attack[i]], 
                                                    self.params['trigger_reduction'], 
                                                    device=device)
        for trigger in self.triggers:
            trigger.eval()
            
    def grad_mask(self, model, participant_ids, criterion, epoch=None):
        """Generate a gradient mask based on the given dataset"""
        if epoch is not None and hasattr(self, "save_epoch") and self.save_epoch == epoch:
            return self.save_grad_mask
        model.train()
        model.apply(fix_bn)
        model.zero_grad()
        ratio = self.params['gradmask_ratio']
        for participant_id in participant_ids:

            _, train_data = self.train_data[participant_id]

            for inputs, labels in train_data:
                inputs, labels = inputs.cuda(), labels.cuda()

                output = model(inputs)

                loss = criterion(output, labels)
                loss.backward()

        mask_grad_list = []
        if self.params['aggregate_all_layer'] == 1:
            grad_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))

            grad_list = torch.cat(grad_list).cuda()
            _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
            mask_flat_all_layer[indices] = 1.0
            
            count = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))

                    mask_flat = mask_flat_all_layer[count:count + gradients_length].cuda()
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                    count += gradients_length
        else:
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    if ratio == 1.0:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                    else:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

        model.zero_grad()
        model.train()
        
        if epoch is not None:
            self.save_grad_mask = mask_grad_list
            self.save_epoch = epoch
            
        return mask_grad_list
    
    def hessian_analysis(self, epoch):
        # we use cuda to make the computation fast
        model = self.target_model.cuda()
        x_list, y_list = [], []
        num_iter = 0
        for batch in self.test_data_poison:
            data, targets, _ = self.get_poison_batch(batch, adversarial_index=-1, evaluation=True)
            x_list.append(data)
            y_list.append(targets)
            if num_iter > 7:
                break
            else:
                num_iter += 1
        x_tensor = torch.cat(x_list)
        y_tensor = torch.cat(y_list)
        # create the hessian computation module
        hessian_comp = hessian(model, torch.nn.CrossEntropyLoss(), data=(x_tensor, y_tensor), cuda=True)

        # Now let's compute the top eigenvalue. This only takes a few seconds.
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
        logger.info("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])

        trace = hessian_comp.trace()
        trace = np.mean(trace)
        logger.info(f'Hessian trace is: {trace}')
        top_eigenvalues_list = [top_eigenvalues]
        trace_list = [trace]
        
        self.save_file(file_name=f'Top_eigenvalue_{epoch}', data_list=top_eigenvalues_list, folder_name=os.path.join(self.folder_path, "Hessian_analysis"))
        self.save_file(file_name=f'Hessian_trace_{epoch}', data_list=trace_list, folder_name=os.path.join(self.folder_path, "Hessian_analysis"))


if __name__ == '__main__':
    np.random.seed(1)
    with open(f'./utils/cifar_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                        name=params_loaded.get('name', 'mnist'))
    helper.load_data()

    pars = list(range(100))
    # show the data distribution among all participants.
    count_all = 0
    for par in pars:
        cifar_class_count = dict()
        for i in range(10):
            cifar_class_count[i] = 0
        count = 0
        _, data_iterator = helper.train_data[par]
        for batch_id, batch in enumerate(data_iterator):
            data, targets= batch
            for t in targets:
                cifar_class_count[t.item()]+=1
            count += len(targets)
        count_all += count
        print(par, cifar_class_count, count, max(zip(cifar_class_count.values(), cifar_class_count.keys())))

    print('avg', count_all*1.0/100)
