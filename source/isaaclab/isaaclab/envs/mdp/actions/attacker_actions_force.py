from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import carb
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import quat_apply
import wandb
# import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg
    
    
class AttackerActionsForce(ActionTerm):
    
    cfg: actions_cfg.AttackerActionsForceCfg
    _asset: Articulation
    _scale: float
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    
    def __init__(self, cfg: actions_cfg.AttackerActionsForceCfg, env: ManagerBasedEnv) -> None:
        
        super().__init__(cfg, env)
        #self.model = torch.jit.load('/home/saka/robot_lab/scripts/rsl_rl/base/logs/rsl_rl/unitree_go2_rough/2025-08-05_19-11-38/exported/policy.pt')#二次攻击对象
        #self.model = torch.jit.load('/home/saka/robot_lab/scripts/rsl_rl/base/logs/rsl_rl/baseline/exported/policy.pt')
        self.model = torch.jit.load('/home/saka/robot_lab/scripts/rsl_rl/base/logs/rsl_rl/loco_asymmetric/2025-07-15_15-45-59_5%/exported/policy.pt') # 非对称
        #self.model = torch.jit.load('/home/saka/robot_lab/scripts/rsl_rl/base/logs/rsl_rl/unitree_go2_rough/2025-06-05_19-11-38/exported/policy.pt')
        self.model.eval()
        self._observation_loco_dim = 45 #235
        
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=self.cfg.preserve_order)
        self._num_joints = len(self._joint_ids)
        
        self._body_ids, self._body_names = self._asset.find_bodies(self.cfg.body_name, preserve_order=self.cfg.preserve_order)
        self._num_bodies = len(self._body_ids) # 
        
        self._action_dim = cfg.action_dim
        
        self._max_force_norm = cfg.max_force if hasattr(cfg, 'max_force') else 20.0
        self._max_command_clip = cfg.max_command if hasattr(cfg, 'max_command') else 0.5
        self._max_obs_norm = cfg.max_obs if hasattr(cfg, 'max_obs_norm') else 0.1
        
        self.update_rate =self.cfg.update_rate if hasattr(cfg, 'update_rate') else 0.05
        self._delta_force =self.update_rate * self._max_force_norm
        self._delta_command = self.update_rate * self._max_command_clip # a litter different 
        self._delta_obs = self.update_rate  * self._max_obs_norm
        
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)        
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)   
        
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device) 
        self._processed_actions = torch.zeros_like(self.raw_actions)
        
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

        self._prev_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_obs = torch.zeros(self.num_envs, 30, device=self.device)
        
        self.visualizer = True
        
        # if self.visualizer:
        #     self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        # else:
        #     self.draw_interface = None
        
        # wandb.init(
        #     project="robotlab",
        #     name="NoiseSize",
        #     config=cfg.to_dict(),
        # )

   
        
    """
    Properties.
    """
    @property
    def observations(self) -> torch.Tensor:
        whole_obs = self._env.observation_manager.compute_whole_obs()
        obs_loco = whole_obs[..., :self._observation_loco_dim]
        return obs_loco 

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions           
    
    @property
    def adv_command(self) -> torch.Tensor:
        return self._processed_actions[:, 3:6] if self.action_dim >= 20 else torch.zeros(3, device=self.device)


    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        if self.action_dim ==3:
            self._raw_actions[:] = actions

            norms_force = torch.norm(self._raw_actions, dim=-1, keepdim=True)
            scaling_force = torch.clamp(self._max_force_norm / norms_force, max=1.0)
            unclamped = self._raw_actions * scaling_force
            
            diff_force = unclamped - self._prev_forces
            
            diff_force = torch.clamp(diff_force, -self._delta_force, self._delta_force)
            
            self._processed_actions = self._prev_forces + diff_force
            self._prev_forces = self._processed_actions.clone()
        
        if self.action_dim >= 20:
            self._raw_actions[:] = actions

            norm_force = torch.norm(self._raw_actions[:,:3], dim=-1, keepdim=True)
            command = self._raw_actions[:,3:6]
            norm_obs = torch.norm(self._raw_actions[:,6:], dim=-1, keepdim=True)
            
            # for forces
            scaling_force = torch.clamp(self._max_force_norm / norm_force, max=1.0)
            unclamped = self._raw_actions[:,:3] * scaling_force
            diff_force = unclamped - self._prev_forces
            diff_force = torch.clamp(diff_force, -self._delta_force, self._delta_force)
            self._processed_actions[:,:3] = self._prev_forces + diff_force
            self._prev_forces = self._processed_actions[:,:3].clone()
            
            # for commands
            clamped_command = torch.clamp(command, -self._max_command_clip, self._max_command_clip)
            diff_command = clamped_command - self._prev_commands
            diff_command = torch.clamp(diff_command, -self._delta_command, self._delta_command)
            self._processed_actions[:,3:6] = self._prev_commands + diff_command
            self._prev_commands = self._processed_actions[:,3:6].clone()
            
            # for observations
            scaling_obs = torch.clamp(self._max_obs_norm / norm_obs, max=1.0)
            unclamped_obs = self._raw_actions[:,6:] * scaling_obs
            diff_obs = unclamped_obs - self._prev_obs
            diff_obs = torch.clamp(diff_obs, -self._delta_obs, self._delta_obs)
            self._processed_actions[:,6:] = self._prev_obs + diff_obs
            self._prev_obs = self._processed_actions[:,6:].clone()
            
    def reset(self, env_ids: Sequence[int] | None = None):
        self._raw_actions[env_ids] = 0.0 
        self._processed_actions[env_ids] = 0.0
        self._prev_forces[env_ids] = 0.0
        self._prev_commands[env_ids] = 0.0
        self._prev_obs[env_ids] = 0.0

        
    def apply_actions(self):

        if self.action_dim == 3:
        
            forces = self._processed_actions.unsqueeze(1)
            
            torques = torch.zeros_like(forces)
             
            forces_draw = forces.clone()
            
            self._asset.set_external_force_and_torque(forces, torques, body_ids=self._body_ids)
            
            # self._draw(forces_draw, self.visualizer)
            
            with torch.no_grad():        
                model = self.model.cuda()
            
                action_loco = model(self.observations)
            
                scaled_action = action_loco * self._scale + self._offset
                
                
                self._asset.set_joint_position_target(scaled_action, joint_ids=self._joint_ids)
                

        if self.action_dim >= 20:
            #
            forces = self._processed_actions[:,:3].unsqueeze(1)
            forces_draw = forces.clone()
            torques = torch.zeros_like(forces)
            self._asset.set_external_force_and_torque(forces, torques, body_ids=self._body_ids)
            # self._draw(forces_draw, self.visualizer)
            # command
            adv_command = self._processed_actions[:,3:6] # for command details, see velocity_command.py

            # obs 
            perturbed = self.observations[:, :self.action_dim-3-3] + self._processed_actions[:, 6:]

            unchanged = self.observations[:, self.action_dim-3-3:]  # unchanged part of the observation
            adv_obs = torch.cat([perturbed, unchanged], dim=1)
            with torch.no_grad():
                model = self.model.cuda()
                action_loco = self.model(adv_obs)
                #print(action_loco)
                scaled_action = action_loco * self._scale + self._offset
                
                self._asset.set_joint_position_target(scaled_action, joint_ids=self._joint_ids)
                


    # def _draw(self, forces_draw: torch.Tensor , visualizer:bool) -> None:
    #     env_ids = forces_draw.shape[0]
        
    #     self.draw_interface.clear_lines()
        
    #     if visualizer:
            
    #         position = self._asset.data.body_pos_w[:, 0, :3].cpu().numpy()

    #         colors = [carb.ColorRgba(0.843, 0.388, 0.392, 1)] * env_ids
            
    #         thinkness = [5.0] * env_ids
            
    #         force_flat_b = forces_draw.squeeze(1).cpu()
            
    #         quat = self._asset.data.body_quat_w[:, 0].cpu()
            
    #         world_forces = quat_apply(quat, force_flat_b)
            
    #         scale = 0.04
            
    #         ends = (position + world_forces.cpu().numpy() * scale).tolist()
            
    #         self.draw_interface.draw_lines(
    #             position.tolist(),
    #             ends,
    #             colors,
    #             thinkness,
    #         )

        

        
        