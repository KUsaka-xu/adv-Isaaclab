# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.utils.assets import check_file_path, read_file
# import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
import random
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

        self.adv_model = torch.jit.load('/home/saka/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/Attack-Force-asymmetric/2025-07-15_22-45-21/exported/policy.pt')#asymmetric
        self.adv_model.eval()
        
        self._body_ids, self._body_names = self._asset.find_bodies(self.cfg.body_name, preserve_order=self.cfg.preserve_order)
        self._num_bodies = len(self._body_ids) #            

        self._max_force_norm = cfg.max_force 
        self._max_command_clip = cfg.max_command 
        self._max_obs_norm = cfg.max_obs  

        self.update_rate =self.cfg.update_rate if hasattr(cfg, 'update_rate') else 0.05
        self._delta_force =self.update_rate * self._max_force_norm
        self._delta_command = self.update_rate * self._max_command_clip # a litter different 
        self._delta_obs = self.update_rate  * self._max_obs_norm 
        
        self._attack_all = torch.zeros(self.num_envs, 3+3+30, device=self.device)
        self._prev_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_obs = torch.zeros(self.num_envs, 30,device=self.device)
        
        self._under_attack_ids = cfg.under_attack_ids
        num_attacks = int(self._under_attack_ids * self.num_envs)
        self._attack_envs = random.sample(range(self.num_envs), num_attacks)
        
        self.visualizer = True
        
        # if self.visualizer:
        #     self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        # else:
        #     self.draw_interface = None
        
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def adv_obs(self):
        """Returns the previous observation."""
        obs_buffer = self._env.observation_manager.compute()
        adv_obs = obs_buffer['rnd_state']
        return adv_obs
    
    @property
    def attack_ids(self):
        """The indices of the environments that are under attack."""
        return self._attack_envs
    
    @property
    def adv_command(self) -> torch.Tensor:
        # 返回 (num_envs, 3) 大小
        return self._attack_all[:, 3:6]
     
    @property
    def perturbed_obs(self) -> torch.Tensor:
        """The perturbed observation tensor."""
        return self._attack_all[:, 6:]
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
            
            
        # for Attack
        adv_obs = self.adv_obs
        with torch.no_grad():
            adv_model = self.adv_model.cuda() 
            self._attack_all =  adv_model(adv_obs)
        
        norm_force = torch.norm(self._attack_all[:, :3],dim = -1, keepdim =True)
        command = self._attack_all[:, 3:6]
        norm_obs = torch.norm(self._attack_all[:, 6:],dim = -1, keepdim =True)
        
        #for force
        scaling_force = torch.clamp(self._max_force_norm / norm_force, max=1.0)
        unclamped = self._attack_all[:, :3] * scaling_force
        diff_force = unclamped - self._prev_forces
        diff_force = torch.clamp(diff_force, -self._delta_force, self._delta_force)
        self._attack_all[:, :3] = self._prev_forces + diff_force
        self._prev_forces = self._attack_all[:, :3].clone()
        
        # for commands
        clamped_command = torch.clamp(command, -self._max_command_clip, self._max_command_clip)
        diff_command = clamped_command - self._prev_commands
        diff_command = torch.clamp(diff_command, -self._delta_command, self._delta_command)
        self._attack_all[:, 3:6] = self._prev_commands + diff_command
        self._prev_commands = self._attack_all[:, 3:6].clone()
        
        # for obs
        scaling_obs = torch.clamp(self._max_obs_norm / norm_obs, max=1.0)
        unclamped_obs = self._attack_all[:, 6:] * scaling_obs
        diff_obs = unclamped_obs - self._prev_obs
        diff_obs = torch.clamp(diff_obs, -self._delta_obs, self._delta_obs)
        self._attack_all[:, 6:] = self._prev_obs + diff_obs
        self._prev_obs = self._attack_all[:, 6:].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._prev_forces[env_ids] = 0.0
        self._prev_commands[env_ids] = 0.0
        self._prev_obs[env_ids] = 0.0

class JointActionsRetrain(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if not check_file_path(cfg.attacker_policy_path):
            raise ValueError(f"Invalid attacker policy path: {cfg.attacker_policy_path}")
        self.adv_model = torch.jit.load(cfg.attacker_policy_path).eval()
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        
        envs_to_attack = self.attack_ids
        forces = self._attack_all[envs_to_attack, :3].unsqueeze(1) # 
        
        # for command details, see velocity_command.py
        
        forces_draw = forces.clone()
        torques = torch.zeros_like(forces)
        self._asset.set_external_force_and_torque(
            forces=forces, torques=torques, body_ids=self._body_ids, env_ids=envs_to_attack,
        )
        # self._draw(forces_draw, self.visualizer)

    # def _draw(self, forces_draw: torch.Tensor, visualizer: bool) -> None:
    #     if not visualizer:
    #         return

       
    #     envs_to_draw = self.attack_ids   # e.g. [12, 345, 589, ...]

    #     self.draw_interface.clear_lines()

    #     starts: list[list[float]] = []
    #     ends:   list[list[float]] = []
    #     colors: list[carb.ColorRgba] = []
    #     thick:  list[float] = []

    #     # 遍历本地 idx 和对应的全局 env idx
    #     for local_idx, global_env in enumerate(envs_to_draw):

    #         pos = self._asset.data.body_pos_w[global_env, 0, :3].cpu().numpy()
    #         quat = self._asset.data.body_quat_w[global_env, 0].cpu()

    #         force_local = forces_draw[local_idx, 0].cpu()  # shape (3,)

    #         world_f = quat_apply(quat, force_local)

    #         end = pos + world_f.numpy() * 0.03

    #         starts.append(pos.tolist())
    #         ends.append(end.tolist())
    #         colors.append(carb.ColorRgba(0.843, 0.388, 0.392, 1.0))
    #         thick.append(5.0)

    #     # 最后一次性画出所有线
    #     self.draw_interface.draw_lines(starts, ends, colors, thick)
