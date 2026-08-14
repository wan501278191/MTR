# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}




def train_one_epoch(model, optimizer, train_loader, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, scheduler=None, show_grad_curve=False,
                    logger=None, logger_iter_interval=50, cur_epoch=None, total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, ema=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    optimizer, optimizer_2 = optimizer if isinstance(optimizer, list) else (optimizer, None)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        optimizer.zero_grad()
        if optimizer_2 is not None:
            optimizer_2.zero_grad()

        loss, tb_dict, disp_dict = model(batch)

        loss.backward()

        total_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)

        optimizer.step()

        if ema is not None:
            ema.update(model)

        if optimizer_2 is not None:
            optimizer_2.step()

        # scheduler.step() MUST be called AFTER optimizer.step()
        if scheduler is not None:
            scheduler.step()

        accumulated_iter += 1
        try:
            cur_lr = optimizer.param_groups[0]['lr']
        except:
            pass
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                trained_time_past_all = tbar.format_dict['elapsed']
                second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                trained_time_each_epoch = pbar.format_dict['elapsed']
                remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
                disp_str += f', lr={disp_dict["lr"]}'
                batch_size = batch.get('batch_size', None)
                logger.info(f'epoch: {cur_epoch}/{total_epochs}, acc_iter={accumulated_iter}, cur_iter={cur_it}/{total_it_each_epoch}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                            f'time_cost(epoch): {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)}, '
                            f'time_cost(all): {tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}, '
                            f'{disp_str}')

            if tb_log is not None:
                # === 1. 模型指标 (model.*) — 总览优先 ===
                tb_log.add_scalar('model.loss', loss.item(), accumulated_iter)
                tb_log.add_scalar('model.learning_rate', cur_lr, accumulated_iter)
                tb_log.add_scalar('model.loss_dense_prediction', tb_dict.get('loss_dense_prediction', 0.0), accumulated_iter)
                # 逐层 loss（先总 loss，再分项）
                for layer_idx in range(6):
                    for sub_tag in [f'loss_layer{layer_idx}', f'loss_layer{layer_idx}_reg_gmm',
                                    f'loss_layer{layer_idx}_reg_vel', f'loss_layer{layer_idx}_cls']:
                        if sub_tag in tb_dict:
                            tb_log.add_scalar(f'model.{sub_tag}', tb_dict[sub_tag], accumulated_iter)

                # === 2. 训练过程 (train.*) — 总览优先 ===
                tb_log.add_scalar('train.total_norm', total_norm, accumulated_iter)
                # ADE 按类别 + 平均
                ade_vals = []
                for obj_type in ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']:
                    ade_key = f'ade_{obj_type}_layer_5'
                    if ade_key in tb_dict:
                        ade_val = tb_dict[ade_key]
                        ade_vals.append(ade_val)
                        tb_log.add_scalar(f'train.ade_{obj_type}', ade_val, accumulated_iter)
                if ade_vals:
                    tb_log.add_scalar('train.ade_avg', sum(ade_vals) / len(ade_vals), accumulated_iter)

                if show_grad_curve:
                    for key, val in model.named_parameters():
                        key = key.replace('.', '/')
                        tb_log.add_scalar('model.grad_' + key, val.grad.abs().max().item(), accumulated_iter)

            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1

    if rank == 0:
        pbar.close()
    return accumulated_iter


def learning_rate_decay(i_epoch, optimizer, optim_cfg):
    if isinstance(optimizer, list):
        optimizer, optimizer_2 = optimizer

    if i_epoch > 0 and i_epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.3

    if optim_cfg.OPTIMIZER == 'complete_traj':
        if i_epoch > 0 and i_epoch % 5 == 0:
            for p in optimizer_2.param_groups:
                p['lr'] *= 0.3


def train_model(model, optimizer, train_loader, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, ckpt_save_dir, train_sampler=None,
                ckpt_save_interval=1, max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, tb_log=None,
                scheduler=None, test_loader=None, logger=None, eval_output_dir=None, cfg=None, dist_train=False,
                logger_iter_interval=50, ckpt_save_time_interval=300):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            torch.cuda.empty_cache()
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            if scheduler is None:
                learning_rate_decay(cur_epoch, optimizer, optim_cfg)

            # train one epoch
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                scheduler=scheduler, cur_epoch=cur_epoch, total_epochs=total_epochs,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if (trained_epoch % ckpt_save_interval == 0 or trained_epoch in [1, 2, 4] or trained_epoch > total_epochs - 10) and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            # eval the model
            if test_loader is not None and (trained_epoch % ckpt_save_interval == 0 or trained_epoch in [1, 2, 4] or trained_epoch > total_epochs - 10):
                from eval_utils.eval_utils import eval_one_epoch

                pure_model = model
                if ema is not None:
                    ema.apply_shadow(model)
                torch.cuda.empty_cache()
                tb_dict = eval_one_epoch(
                    cfg, pure_model, test_loader, epoch_id=trained_epoch, logger=logger, dist_test=dist_train,
                    result_dir=eval_output_dir, save_to_file=False, logger_iter_interval=max(logger_iter_interval // 5, 1)
                )
                if ema is not None:
                    ema.restore(model)
                if cfg.LOCAL_RANK == 0:
                    # === 3. 评估结果 (eval.*) — 所有时间点总览优先，再分时间点细项 ===
                    ordered_keys = []
                    # Phase 1: all avg summaries first (3s/5s/8s × 4 metrics = 16 keys)
                    for es in ['3s', '5s', '8s']:
                        for metric in ['mAP', 'minADE', 'minFDE', 'MissRate']:
                            k = f'{es}_{metric}'
                            if k in tb_dict:
                                ordered_keys.append(k)
                    # Phase 2: per-time-window per-type breakdown
                    for es in ['3s', '5s', '8s']:
                        for metric in ['mAP', 'minADE', 'minFDE', 'MissRate']:
                            for obj_type in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
                                k = f'{es}_{metric} - {obj_type}'
                                if k in tb_dict:
                                    ordered_keys.append(k)
                    # Phase 3: backward-compat un-prefixed (8s default)
                    for key in tb_dict:
                        if key not in ordered_keys and isinstance(tb_dict[key], (int, float)):
                            ordered_keys.append(key)

                    for key in ordered_keys:
                        val = tb_dict[key]
                        # Build SwanLab tag with '/' group separator.
                        # Avg metrics get '0_summary/' subgroup to sort before per-type metrics.
                        if key[:3] in ['3s_', '5s_', '8s_']:
                            prefix = key[:2]  # '3s', '5s', '8s'
                            rest = key[3:]     # 'mAP' or 'mAP - VEHICLE'
                            rest = rest.replace(' - ', '_').replace(' ', '')
                            if not any(t in rest for t in ['_VEHICLE', '_PEDESTRIAN', '_CYCLIST']):
                                # Avg summary: eval/0_summary/3s.mAP, eval/0_summary/5s.mAP, etc.
                                tag = f'eval/0_summary/{prefix}.{rest}'
                            else:
                                # Per-type: eval/3s.mAP_VEHICLE, eval/5s.minADE_PEDESTRIAN, etc.
                                tag = f'eval/{prefix}.{rest}'
                        elif ' - ' in key:
                            rest = key.replace(' - ', '_').replace(' ', '')
                            tag = f'eval/{rest}'
                        else:
                            tag = f'eval/{key}'
                        tb_log.add_scalar(tag, val, trained_epoch)

                    if 'mAP' in tb_dict:
                        # Composite score: prioritize ADE/FDE (lower=better), then mAP (higher=better)
                        # score = mAP - ADE - FDE  (higher is better)
                        composite_score = tb_dict.get('mAP', 0) - tb_dict.get('minADE', 0) - tb_dict.get('minFDE', 0)
                        best_record_file = eval_output_dir / ('best_eval_record.txt')

                        try:
                            with open(best_record_file, 'r') as f:
                                best_src_data = f.readlines()
                            # Parse: best_epoch_xx score -x.xxx
                            best_performance = float(best_src_data[-1].strip().split(' ')[-1])
                        except:
                            with open(best_record_file, 'a') as f:
                                pass
                            best_performance = -999.0

                        with open(best_record_file, 'a') as f:
                            print(f'epoch_{trained_epoch} score {composite_score:.4f} mAP {tb_dict.get("mAP", 0):.4f} ADE {tb_dict.get("minADE", 0):.4f} FDE {tb_dict.get("minFDE", 0):.4f}', file=f)

                        if composite_score > best_performance:
                            ckpt_name = ckpt_save_dir / 'best_model'
                            save_checkpoint(
                                checkpoint_state(model, epoch=cur_epoch, it=accumulated_iter), filename=ckpt_name,
                            )
                            logger.info(f'Save best model (score={composite_score:.4f}, mAP={tb_dict.get("mAP",0):.4f}, ADE={tb_dict.get("minADE",0):.4f}, FDE={tb_dict.get("minFDE",0):.4f}) to {ckpt_name}')

                            with open(best_record_file, 'a') as f:
                                print(f'best_epoch_{trained_epoch} score {composite_score:.4f}', file=f)
                        else:
                            with open(best_record_file, 'a') as f:
                                print(f'{best_src_data[-1].strip()}', file=f)
                    else:
                        raise NotImplementedError


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import mtr
        version = 'mtr+' + mtr.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
