import copy
import sys

from torch.autograd import Variable
import numpy as np
import torch
import cv2
    

class FeatureExtractor():
    """ Class for extracting activations and registering gradients from targetted intermediate layers.
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def __call__(self, x):
        
        target_activations = []
        self.gradients = []
        
        module_tag_list = set(copy.deepcopy(self.target_layers))
        NO_MONITOR = 0
        GETATTR_CALL = 1
        GETATTR_RETURN = 2
        MODULE_CALL = 3
        stage = NO_MONITOR
        module_hash = None
        def trace_calls(frame, event, arg = None):
            if event not in ['call', 'return']:
                return trace_calls
            
            code = frame.f_code
            func_name = code.co_name
            nonlocal stage, module_hash, target_activations
            if stage == NO_MONITOR:
                if func_name == '__getattr__':
                    name = frame.f_locals['name']
                    if name in module_tag_list:
                        stage = GETATTR_CALL
                        module_tag_list.remove(name)
            elif stage == GETATTR_CALL:
                if func_name == '__getattr__' and event == 'return':
                    stage = GETATTR_RETURN
            elif stage == GETATTR_RETURN:
                stage = MODULE_CALL
                module_hash = frame.__hash__()
            elif stage == MODULE_CALL:
                if event == 'return' and module_hash == frame.__hash__():
                    x = frame.f_locals.get('result', None)
                    x.register_hook(self.save_gradient)
                    target_activations += [x]
                    stage = NO_MONITOR
            
            return trace_calls
    
        sys.settrace(trace_calls)  # Set custom trace function
        output = self.model.forward(x)  # Execute the function 'fuc'
        sys.settrace(None)  # Disable custom trace function
        return target_activations, output
    

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targetted layers.
    3. Gradients from intermediate targetted layers. """
    def __init__(self, model, target_layers=None):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        return target_activations, output


class GradCam:
    def __init__(self, model, device, target_layers=None, size=(32, 32)):
        self.model = model
        self.model.eval()
        self.model = model.to(device)
        self.device = device
        self.extractor = ModelOutputs(self.model, target_layers)
        self.size = size

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        input = input.to(self.device)
        
        features, output = self.extractor(input)
            
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True).to(self.device)

        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[0].cpu().data.numpy()   # 1 x 3 x 32 x 32 
        target = features[-1]   # 1 x 3 x 32 x 32
        target = target.cpu().data.numpy()[0, :]    # 3 x 32 x 32
        weights = np.mean(grads_val, axis = (2, 3))[0, :]   # 3
        
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)  # 32 x 32
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)    # keep non-negative
        cam = cv2.resize(cam, self.size)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1.e-8)
        return cam, index
