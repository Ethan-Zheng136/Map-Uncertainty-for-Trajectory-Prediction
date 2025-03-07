# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import GlobalInteractor
from models import MLPDecoder
from utils import TemporalData

import os
import pickle

class VotingMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2, temperature=0.1): 
        super(VotingMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )
        self.temperature = temperature
        self.target_length = 1920

    def temperature_scaled_softmax(self, logits):
        scaled_logits = logits / self.temperature
        exp_logits = torch.exp(scaled_logits)
        softmax_output = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
        
        return softmax_output

    def forward(self, y_hat_unc, y_hat_bev):
        y_hat_unc_flat = y_hat_unc.flatten()
        y_hat_bev_flat = y_hat_bev.flatten()
        y_hat_unc_padded = F.pad(y_hat_unc_flat, (0, self.target_length - y_hat_unc_flat.size(0)), "constant", 0)
        y_hat_bev_padded = F.pad(y_hat_bev_flat, (0, self.target_length - y_hat_bev_flat.size(0)), "constant", 0)
        mlp_input = torch.cat([y_hat_unc_padded, y_hat_bev_padded], dim=-1)

        intermediate_output = self.mlp[:-1](mlp_input.unsqueeze(0)) 

        logits = self.mlp(mlp_input.unsqueeze(0))

        weights = self.temperature_scaled_softmax(logits).squeeze(0)

        return weights, logits

class HiVT(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 temperature: float,
                 method=None,
                 map_model=None,
                 **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters(ignore=['method'])

        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.temperature = temperature
        self.voting_mlp  = VotingMLP(input_dim=3840, temperature=self.temperature).to(self.device)
        
        from models.local_encoder import LocalEncoder as BaseLocalEncoder
        from models.local_encoder_bev import LocalEncoder as BevLocalEncoder
        from models.local_encoder_std import LocalEncoder as UncLocalEncoder
        
        
        
        self.local_encoder_unc = BaseLocalEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    num_temporal_layers=num_temporal_layers,
                                    local_radius=local_radius,
                                    parallel=parallel)
    
        
        self.local_encoder_bev = BevLocalEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    num_temporal_layers=num_temporal_layers,
                                    local_radius=local_radius,
                                    parallel=parallel,
                                    map_model=map_model)

        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()

    def temperature_scaled_softmax(self, logits, temperature=0.5):
        scaled_logits = logits / temperature
        exp_logits = torch.exp(scaled_logits)
        softmax_output = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
        
        return softmax_output

    def forward(self, data: TemporalData):
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        local_embed_unc = self.local_encoder_unc(data=data)
        local_embed_bev = self.local_encoder_bev(data=data)
        global_embed_unc = self.global_interactor(data=data, local_embed=local_embed_unc)
        global_embed_bev = self.global_interactor(data=data, local_embed=local_embed_bev)
        y_hat_unc, pi_unc = self.decoder(local_embed=local_embed_unc, global_embed=global_embed_unc)
        y_hat_bev, pi_bev = self.decoder(local_embed=local_embed_bev, global_embed=global_embed_bev)
        
        return y_hat_unc, pi_unc, y_hat_bev, pi_bev


    def training_step(self, data, batch_idx):
        torch.cuda.empty_cache()  
        self.voting_mlp = self.voting_mlp.to(self.device)
        y_hat_unc, pi_unc, y_hat_bev, pi_bev = self(data)

        ade_unc = ADE().to(self.device)
        fde_unc = FDE().to(self.device)
        mr_unc= MR().to(self.device)
        
        ade_bev = ADE().to(self.device)
        fde_bev = FDE().to(self.device)
        mr_bev = MR().to(self.device)

        y_agent = data.y[data['av_index']]
        y_hat_agent_unc = y_hat_unc[:, data['av_index'], :, : 2]
        y_hat_agent_bev = y_hat_bev[:, data['av_index'], :, : 2]

        fde_agent_unc = torch.norm(y_hat_agent_unc[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        fde_agent_bev = torch.norm(y_hat_agent_bev[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent_unc = fde_agent_unc.argmin(dim=0)
        best_mode_agent_bev = fde_agent_bev.argmin(dim=0)
        y_hat_best_agent_unc = y_hat_agent_unc[best_mode_agent_unc, torch.arange(data.num_graphs)]
        y_hat_best_agent_bev = y_hat_agent_bev[best_mode_agent_bev, torch.arange(data.num_graphs)]

        ade_unc.update(y_hat_best_agent_unc.to(self.device), y_agent.to(self.device))
        fde_unc.update(y_hat_best_agent_unc.to(self.device), y_agent.to(self.device))
        mr_unc.update(y_hat_best_agent_unc.to(self.device), y_agent.to(self.device))

        ade_bev.update(y_hat_best_agent_bev.to(self.device), y_agent.to(self.device))
        fde_bev.update(y_hat_best_agent_bev.to(self.device), y_agent.to(self.device))
        mr_bev.update(y_hat_best_agent_bev.to(self.device), y_agent.to(self.device))

        votes_unc = 0
        votes_bev = 0
        
        # no tie, if so, let bev go
        if ade_unc.compute() < ade_bev.compute():
            votes_unc += 1
        elif ade_unc.compute() >= ade_bev.compute():
            votes_bev += 1
        
        if fde_unc.compute() < fde_bev.compute():
            votes_unc += 1
        elif fde_unc.compute() >= fde_bev.compute():
            votes_bev += 1
        
        if mr_unc.compute() < mr_bev.compute():
            votes_unc += 1
        elif mr_unc.compute() >= mr_bev.compute():
            votes_bev += 1

        logit_unc = torch.tensor(votes_unc, dtype=torch.float32, device=self.device)
        logit_bev = torch.tensor(votes_bev, dtype=torch.float32, device=self.device)
        votes_true = self.temperature_scaled_softmax(torch.stack([logit_unc, logit_bev]))
        

        
        weights, logits = self.voting_mlp(y_hat_best_agent_unc, y_hat_best_agent_bev)
        
        mse_loss = F.mse_loss(weights, votes_true)
       
        y_hat_combined = weights[0] * y_hat_unc + weights[1] * y_hat_bev  
        pi_combined = weights[0] * pi_unc + weights[1] * pi_bev


        reg_mask = ~data['padding_mask'][:, self.historical_steps:]       
        valid_steps = reg_mask.sum(dim=-1)                                
        cls_mask = valid_steps > 0                                        
        l2_norm = (torch.norm(y_hat_combined[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  
        best_mode = l2_norm.argmin(dim=0)                                 
        y_hat_best = y_hat_combined[best_mode, torch.arange(data.num_nodes)]    
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()      
        cls_loss = self.cls_loss(pi_combined[cls_mask], soft_target)
        loss = reg_loss + cls_loss + mse_loss
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=32)
        self.log('train_cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=32)
        self.log('train_mse_loss', mse_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=32)
        self.log('train_total_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=32)
        
        return loss


    def validation_step(self, data, batch_idx):
        y_hat_unc, pi_unc, y_hat_bev, pi_bev = self(data)
        y_agent = data.y[data['av_index']]
        y_hat_agent_unc = y_hat_unc[:, data['av_index'], :, : 2]
        y_hat_agent_bev = y_hat_bev[:, data['av_index'], :, : 2]
        # print('y_hat_unc:', y_hat_unc.shape)
        # print('y_hat_bev:', y_hat_bev.shape)
        # print('----------------------------------------------------------------------------------------------------')
        fde_agent_unc = torch.norm(y_hat_agent_unc[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        fde_agent_bev = torch.norm(y_hat_agent_bev[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent_unc = fde_agent_unc.argmin(dim=0)
        best_mode_agent_bev = fde_agent_bev.argmin(dim=0)
        y_hat_best_agent_unc = y_hat_agent_unc[best_mode_agent_unc, torch.arange(data.num_graphs)]
        y_hat_best_agent_bev = y_hat_agent_bev[best_mode_agent_bev, torch.arange(data.num_graphs)]
 
        with torch.no_grad():  
            weights, logits = self.voting_mlp(y_hat_best_agent_unc, y_hat_best_agent_bev)

        y_hat_combined = weights[0] * y_hat_unc + weights[1] * y_hat_bev
        pi_combined = weights[0] * pi_unc + weights[1] * pi_bev

        # self.visualization(data, batch_idx, y_hat_combined, pi_combined)
        # global_y_hat_agent, pi_agent, seq_id = self.visualization(data, batch_idx)

        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat_combined[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat_combined[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = y_hat_combined[:, data['av_index'], :, : 2]
        y_agent = data.y[data['av_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
    

    def visualization(self, data, batch_idx):
        data = data.clone()
        y_hat, pi = self(data)
        pi = F.softmax(pi)
        y_hat = y_hat.permute(1, 0, 2, 3)
        y_hat_agent = y_hat[data['av_index'], :, :, :2]
        y_hat_agent_uncertainty = y_hat[data['av_index'], :, :, 2:4]
        pi_agent = pi[data['av_index'], :]
        if self.rotate:
            data_angles = data['theta']
            data_origin = data['origin']
            data_rotate_angle = data['rotate_angles'][data['av_index']]
            data_local_origin = data.positions[data['av_index'], 19, :]
            rotate_mat = torch.empty(len(data['av_index']), 2, 2, device=self.device)
            sin_vals = torch.sin(-data_angles)
            cos_vals = torch.cos(-data_angles)
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            rotate_local = torch.empty(len(data['av_index']), 2, 2, device=self.device)
            sin_vals_angle = torch.sin(-data_rotate_angle)
            cos_vals_angle = torch.cos(-data_rotate_angle)
            rotate_local[:, 0, 0] = cos_vals_angle
            rotate_local[:, 0, 1] = -sin_vals_angle
            rotate_local[:, 1, 0] = sin_vals_angle
            rotate_local[:, 1, 1] = cos_vals_angle
            for i in range(len(data['av_index'])):
                stacked_rotate_mat = torch.stack([rotate_mat[i]] * self.num_modes, dim=0)
                stacked_rotate_local = torch.stack([rotate_local[i]] * self.num_modes, dim=0)

                y_hat_agent[i, :, :, :] = torch.bmm(y_hat_agent[i, :, :, :], stacked_rotate_local) \
                                          + data_local_origin[i].unsqueeze(0).unsqueeze(0)
                y_hat_agent[i, :, :, :] = torch.bmm(y_hat_agent[i, :, :, :], stacked_rotate_mat) \
                                          + data_origin[i].unsqueeze(0).unsqueeze(0)
        
        save_path = 'result.pkl'
        if os.path.exists(save_path):
            with open(save_path, 'rb') as file:
                predict_data = pickle.load(file)
            for i in range(len(data['seq_id'])):
                predict_data[data['seq_id'][i].item()] = torch.cat([y_hat_agent[i], y_hat_agent_uncertainty[i]], dim=-1).cpu().numpy()
        else:
            predict_data = {}
            for i in range(len(data['seq_id'])):
                predict_data[data['seq_id'][i].item()] = torch.cat([y_hat_agent[i], y_hat_agent_uncertainty[i]], dim=-1).cpu().numpy()
        with open(save_path, 'wb') as file:
            pickle.dump(predict_data, file)

        return y_hat_agent, pi_agent, data['seq_id']

    def visualization_all(self, data, batch_idx):
        data = data.clone()
        y_hat, pi = self(data)
        pi = F.softmax(pi)
        y_hat = y_hat.permute(1, 0, 2, 3)
        y_hat_agent = y_hat[:, :, :, :2]
        y_hat_agent_uncertainty = y_hat[:, :, :, 2:4]
        pi_agent = pi[:, :]
        if self.rotate:
            data_angles = data['theta']
            data_origin = data['origin']
            av_index = data['av_index']
            av_index = torch.cat((av_index, torch.tensor([data['positions'].shape[0]], device=av_index.device, dtype=av_index.dtype)))
            replication_counts = (av_index[1:] - av_index[:-1]).cpu()
            replicated_data_angles = torch.cat([data_angles[i].repeat(count) for i, count in enumerate(replication_counts)])
            replicated_data_origin = torch.cat([data_origin[i].repeat(count, 1) for i, count in enumerate(replication_counts)])

            data_rotate_angle = data['rotate_angles']
            data_local_origin = data.positions[:, 19, :]
            rotate_mat = torch.empty(y_hat.shape[0], 2, 2, device=self.device)
            sin_vals = torch.sin(-replicated_data_angles)
            cos_vals = torch.cos(-replicated_data_angles)
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            rotate_local = torch.empty(y_hat.shape[0], 2, 2, device=self.device)
            sin_vals_angle = torch.sin(-data_rotate_angle)
            cos_vals_angle = torch.cos(-data_rotate_angle)
            rotate_local[:, 0, 0] = cos_vals_angle
            rotate_local[:, 0, 1] = -sin_vals_angle
            rotate_local[:, 1, 0] = sin_vals_angle
            rotate_local[:, 1, 1] = cos_vals_angle
            for i in range(y_hat.shape[0]):
                stacked_rotate_mat = torch.stack([rotate_mat[i]] * self.num_modes, dim=0)
                stacked_rotate_local = torch.stack([rotate_local[i]] * self.num_modes, dim=0)
                y_hat_agent[i, :, :, :] = torch.bmm(y_hat_agent[i, :, :, :], stacked_rotate_local) \
                                          + data_local_origin[i].unsqueeze(0).unsqueeze(0)
                y_hat_agent[i, :, :, :] = torch.bmm(y_hat_agent[i, :, :, :], stacked_rotate_mat) \
                                          + replicated_data_origin[i].unsqueeze(0).unsqueeze(0)

        save_path = 'result.pkl'
        cumsum_count = replication_counts.cumsum(dim=0)
        if os.path.exists(save_path):
            with open(save_path, 'rb') as file:
                predict_data = pickle.load(file)
        else:
            predict_data = {}
            
        traj_list = []
        scene_idx = 0  # current scene index
        
        for i in range(data['positions'].shape[0]):  # loop through all 492 vehicles
            # Add current trajectory
            traj_list.append(torch.cat([y_hat_agent[i], y_hat_agent_uncertainty[i]], dim=-1).cpu())
            
            # Check if we've reached the end of current scene
            if i + 1 == cumsum_count[scene_idx]:
                # Save current scene's trajectories
                predict_data[data['seq_id'][scene_idx].item()] = torch.cat([v.unsqueeze(0) for v in traj_list], dim=0).numpy()
                traj_list = []
                scene_idx += 1

        with open(save_path, 'wb') as file:
            pickle.dump(predict_data, file)

        return y_hat_agent, pi_agent, data['seq_id']

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2) 
        parser.add_argument('--edge_dim', type=int, default=2) 
        parser.add_argument('--embed_dim', type=int, required=True)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.2) 
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)  
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
