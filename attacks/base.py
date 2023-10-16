from __future__ import annotations

from typing import List

import torch


class Attack(torch.nn.Module):
    """Abstract class to define attack.
    """

    def __init__(self):
        super().__init__()

    def forward(self, images: torch.Tensor) -> torch.Tensor: 
        """Add the trigger to images.

        Args:
            images (torch.tensor): images to be added trigger.
            
        Returns:
            torch.Tensor: images with trigger.
        """
        raise NotImplementedError

    def get_pattern(self) -> torch.Tensor:
        """Get pattern of trigger.

        Returns:
            torch.Tensor: pattern of trigger.
        """
        raise NotImplementedError

    @torch.no_grad()
    def set_by_combination_(self, triggers: List[Attack] | None, reduction: str = "add", device: torch.device = None) -> None:
        """Set up by merging triggers.

        Args:
            triggers (List[Attack] | None): triggers to set up.
            reduction (str, optional): method of reduction. Defaults to "add".
            device (torch.device, optional): device of tensor. Defaults to None.
        """
        raise NotImplementedError
