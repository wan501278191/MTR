# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .context_encoder import build_context_encoder
from .motion_decoder import build_motion_decoder


class MotionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER)
        self.motion_decoder = build_motion_decoder(
            in_channels=self.context_encoder.num_out_channels,
            config=self.model_cfg.MOTION_DECODER
        )

        # === Optional: Diffusion trajectory refiner (from DiffSemanticFusion) ===
        self.use_diffusion_refiner = self.model_cfg.MOTION_DECODER.get('USE_DIFFUSION_REFINER', False)
        if self.use_diffusion_refiner:
            from mtr.models.diffusion import TrajectoryDiffusionRefiner
            diff_cfg = self.model_cfg.MOTION_DECODER.get('DIFFUSION_CONFIG', {})
            # cond_dim must match center_objects_feature dimension (context encoder's D_MODEL)
            self.diffusion_refiner = TrajectoryDiffusionRefiner(
                traj_dim=diff_cfg.get('TRAJ_DIM', 2),
                cond_dim=self.model_cfg.CONTEXT_ENCODER.D_MODEL,
                hidden_dim=diff_cfg.get('HIDDEN_DIM', 128),
                num_diffusion_steps=diff_cfg.get('NUM_STEPS', 20),
            )
            self.diffusion_loss_weight = diff_cfg.get('LOSS_WEIGHT', 0.1)
            self.diffusion_refinement_weight = diff_cfg.get('REFINEMENT_WEIGHT', 0.3)
            # Refiner is trained jointly but is NOT applied at inference by default:
            # the well-trained MTR prediction is sharper than the weak diffusion reconstruction.
            # Set APPLY_REFINER_AT_INFER: True to enable inference refinement.
            self.apply_refiner_at_infer = diff_cfg.get('APPLY_REFINER_AT_INFER', False)

    def forward(self, batch_dict):
        batch_dict = self.context_encoder(batch_dict)
        batch_dict = self.motion_decoder(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_loss()

            # Diffusion refiner training loss
            if self.use_diffusion_refiner:
                # Use GT trajectory as target, center object feature as condition
                gt_trajs = batch_dict['input_dict']['center_gt_trajs'][:, :, 0:2]  # (N, T, 2)
                # Use the center object feature as conditioning
                cond_feat = batch_dict['center_objects_feature']  # (N, D)
                diff_loss = self.diffusion_refiner.compute_loss(gt_trajs, cond_feat)
                loss = loss + self.diffusion_loss_weight * diff_loss
                tb_dict['diffusion_loss'] = diff_loss.item()

            tb_dict.update({'loss': loss.item()})
            disp_dict.update({'loss': loss.item()})
            return loss, tb_dict, disp_dict

        # Diffusion trajectory refinement at inference (disabled by default;
        # the refiner tends to degrade the sharper MTR prediction — enable only if validated)
        if self.use_diffusion_refiner and self.apply_refiner_at_infer and 'pred_trajs' in batch_dict:
            pred_trajs = batch_dict['pred_trajs']  # (N, 6, T, 7)
            pred_scores = batch_dict['pred_scores']  # (N, 6)
            cond_feat = batch_dict['center_objects_feature']  # (N, D)

            # Refine the best-scoring mode for each center object
            best_idx = pred_scores.argmax(dim=-1)  # (N,)
            N = pred_trajs.shape[0]
            best_trajs = pred_trajs[torch.arange(N), best_idx, :, 0:2]  # (N, T, 2)

            refined_trajs = self.diffusion_refiner.refine_trajectory(
                best_trajs, cond_feat, refinement_weight=self.diffusion_refinement_weight
            )

            # Replace best mode's xy with refined version
            pred_trajs[torch.arange(N), best_idx, :, 0:2] = refined_trajs
            batch_dict['pred_trajs'] = pred_trajs

        return batch_dict

    def get_loss(self):
        loss, tb_dict, disp_dict = self.motion_decoder.get_loss()

        return loss, tb_dict, disp_dict

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        # Strip DDP "module." prefix if present (DDP-saved checkpoints)
        model_state = checkpoint['model_state']
        model_state = {k[7:] if k.startswith('module.') else k: v for k, v in model_state.items()}
        self.load_state_dict(model_state, strict=True)

        if optimizer is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        # logger.info('==> Done')
        logger.info('==> Done (loaded %d/%d)' % (len(checkpoint['model_state']), len(checkpoint['model_state'])))

        return it, epoch

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        model_state = self.state_dict()

        # Strip DDP "module." prefix if present (DDP/EMA checkpoints may have it)
        model_state_disk = {
            (k[7:] if k.startswith('module.') else k): v
            for k, v in model_state_disk.items()
        }

        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)

        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'The number of missing keys: {len(missing_keys)}')
        logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        logger.info('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch


