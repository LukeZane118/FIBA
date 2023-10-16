from __future__ import annotations

from typing import List

import torch

from .base import Attack


class DBA(Attack):
    """DBA method to add trigger.
    """
    def __init__(self, patterns: List[List] | None = [], image_shape: tuple = (3, 32, 32)):
        """
        Args:
            patterns (List[List] | None, optional): indices of pixel to adjust. E.g., [[x1, y1], [x2, y2], ...]. 
        """        
        super().__init__()
        self.image_shape = image_shape
        if patterns is not None and len(patterns) > 0:
            self.patterns_x, self.patterns_y = list(map(list, zip(*patterns)))
        else:
            self.patterns_x, self.patterns_y = [], []

    def forward(self, images: torch.Tensor) -> torch.Tensor: 
        """Add the trigger to images.

        Args:
            images (torch.Tensor): images to be added trigger.

        Returns:
            torch.Tensor: images with trigger.
        """ 
        assert 2 <= len(images.shape) <= 4
        
        if images.dim() == 4:
            images[:, :, self.patterns_x, self.patterns_y] = 1
        elif images.dim() == 3:
            images[:, self.patterns_x, self.patterns_y] = 1
        elif images.dim() == 2:
            images[self.patterns_x, self.patterns_y] = 1
        
        return images

    def get_pattern(self) -> torch.Tensor:
        """Get pattern of trigger.

        Returns:
            torch.Tensor: pattern of trigger.
        """
        pattern = torch.zeros(self.image_shape)
        pattern[:, self.patterns_x, self.patterns_y] = 1
        return pattern

    @torch.no_grad()
    def set_by_combination_(self, triggers: List[DBA] | None, reduction: str = "add", device: torch.device = None) -> None:
        """Set up by merging triggers.

        Args:
            triggers (List[DBA] | None): triggers to set up.
            reduction (str, optional): method of reduction. Defaults to "add".
            device (torch.device, optional): device of tensor. Defaults to None.
        """
        if triggers is None or len(triggers) == 0:
            return False
        
        self.image_shape = triggers[0].image_shape
        self.patterns_x = sum((trigger.patterns_x for trigger in triggers), [])
        self.patterns_y = sum((trigger.patterns_y for trigger in triggers), [])
            
        return True
