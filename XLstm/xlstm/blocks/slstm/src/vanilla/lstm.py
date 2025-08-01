# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbininan Pöppel
from typing import Tuple, Dict

import torch


def slstm_forward_pointwise(
    Wx: torch.Tensor,  # dim [B, 4*H]
    Ry: torch.Tensor,  # dim [B, 4*H]
    b: torch.Tensor,   # dim [1, 4*H]
    states: torch.Tensor,  # dim [4, B, H]
    constants: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    _ = constants
    raw = Wx + Ry + b
    _, c = torch.unbind(states.view(2, states.shape[1], -1), dim=0)
    # raw = raw.view(-1, 4, -1)
    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)
    # with torch.no_grad():  # THE difference to maxg aka max_gradient (here max / max_static)
    ogate = torch.sigmoid(oraw)
    igate = torch.sigmoid(iraw)
    fgate = torch.sigmoid(fraw)
    zval = torch.tanh(zraw)
    cnew = fgate * c + igate * zval
    ynew = ogate * torch.tanh(cnew)

    # shapes ([B,H], [B,H], [B,H]), ([B,H],[B,H],[B,H],[B,H])
    return torch.stack((ynew, cnew), dim=0), torch.stack(
        (igate, fgate, zraw, ogate), dim=0
    )
