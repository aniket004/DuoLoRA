from typing import Optional, Union
import torch
from torch import nn
import numpy as np
import torch.cuda.amp as amp

class ZipLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_merger_value: Optional[float] = 1.0,
        init_merger_value_2: Optional[float] = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
                
        self.up_1 = nn.Parameter(
            torch.zeros((out_features, 64), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.up_2 = nn.Parameter(
            torch.zeros((out_features, 64), device=device, dtype=dtype),
            requires_grad=False,
        )
        
        self.down_1 = nn.Parameter(
            torch.zeros(( 64, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.down_2 = nn.Parameter(
            torch.zeros(( 64, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )

        self.merger_1 = nn.Parameter(init_merger_value)
        self.merger_2 = nn.Parameter(init_merger_value_2)
        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"

    def set_forward_type(self, type: str = "merge"):
        assert type in ["merge", "weight_1", "weight_2"]
        self.forward_type = type

    def compute_mergers_similarity(self):
        #breakpoint()
        return (self.merger_1 * self.merger_2).abs().mean()
    
    def merger_rank_minimization_content(self, nuclear_norm_coeff, sparsity_coeff):
        U1, S1, V1 = torch.svd(self.merger_1)
        U2, S2, V2 = torch.svd(self.merger_2)
        nuclear_norm = torch.sum(S2) - torch.sum(S1)
        # print('nuclear norm', nuclear_norm)
        # print('l1_m1',  torch.sum((self.merger_1).abs()))
        # print('l1_m2',  torch.sum((self.merger_2).abs()))
        loss = nuclear_norm_coeff * nuclear_norm + sparsity_coeff * torch.sum((self.merger_1).abs())
        #loss = nuclear_norm + 1*torch.sum((self.merger_1).abs())
        return loss 
    
    def merger_rank_minimization_style(self, nuclear_norm_coeff, sparsity_coeff):
        U1, S1, V1 = torch.svd(self.merger_1)
        U2, S2, V2 = torch.svd(self.merger_2)
        nuclear_norm = torch.sum(S1) - torch.sum(S2)
        # print('nuclear norm', nuclear_norm)
        # print('l1_m1',  torch.sum((self.merger_1).abs()))
        # print('l1_m2',  torch.sum((self.merger_2).abs()))
        loss = nuclear_norm_coeff * nuclear_norm + sparsity_coeff * torch.sum((self.merger_2).abs())
        #loss = nuclear_norm + 1*torch.sum((self.merger_2).abs())
        return loss 
          
    
    def get_ziplora_weight1(self):
        mat1 = (self.merger_1 * self.down_1)
        mat = self.up_1 @ mat1

        return mat

    
    def get_ziplora_weight2(self):
        mat1 = (self.merger_2 * self.down_2)
        mat = self.up_2 @ mat1

        return mat

    
    #import torch.cuda.amp as amp
    def get_ziplora_weight(self):
        with amp.autocast():
            part_1 = self.merger_1 * self.down_1
            part_2 = self.up_1 @ part_1
            part_3 = self.merger_2 * self.down_2
            part_4 = self.up_2 @ part_3

            return part_2 + part_4
    

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.up_1.dtype
        if self.forward_type == "merge":
            weight = self.get_ziplora_weight()
        elif self.forward_type == "weight_1":
            weight = self.up_1 @ self.down_1
        elif self.forward_type == "weight_2":
            weight = self.up_2 @ self.down_2
        else:
            raise ValueError(self.forward_type)
        hidden_states = nn.functional.linear(hidden_states.to(dtype), weight=weight)
        return hidden_states.to(orig_dtype)


class ZipLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)
