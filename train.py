import image_train

def train(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys):
    epochs_submit_update_dict={}
    num_samples_dict={}
    epochs_submit_update_dict, num_samples_dict = image_train.ImageTrain(helper, start_epoch, local_model,
                                                                            target_model, is_poison, agent_name_keys)
    return epochs_submit_update_dict, num_samples_dict
