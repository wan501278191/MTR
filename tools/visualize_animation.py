#!/usr/bin/env python3
"""
Animate MTR trajectory predictions as a GIF showing future N seconds.

Each frame advances 0.1s (10Hz), drawing:
  - Map polylines (static)
  - Historical trajectory (blue, grows frame-by-frame)
  - Agent position marker (red dot, moves forward)
  - Predicted trajectories (6 modes, opacity ∝ score, up to current frame)
  - Ground-truth future (black, up to current frame)

Usage:
    python visualize_animation.py \
        --result_pkl ../output/waymo/mtr_voyah_smoke/local_smoke/eval/epoch_1/smoke_testA/result.pkl \
        --data_dir ../../data/processed_scenarios_testing_A_full \
        --scenario_id enc_00ab2662a208ecc4a140e9a9 \
        --output_dir ../output/visualization \
        --future_seconds 8
"""
import _init_path
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Type mappings ─────────────────────────────────────────────────────────
LANE_TYPES = {1, 2, 3}
ROAD_LINE_TYPES = {6, 7, 8, 9, 10, 11, 12, 13}
ROAD_EDGE_TYPES = {15, 16}
CROSSWALK_TYPE = 18
STOP_SIGN_TYPE = 17
SPEED_BUMP_TYPE = 19

MAP_STYLE = {
    'lane':       {'color': 'green',  'lw': 0.5, 'alpha': 0.5},
    'road_line':  {'color': 'yellow', 'lw': 0.5, 'alpha': 0.6},
    'road_edge':  {'color': 'red',    'lw': 0.8, 'alpha': 0.6},
    'crosswalk':  {'color': 'gray',   'lw': 0.5, 'alpha': 0.3},
}

DT = 0.1  # 10 Hz
HISTORY_LEN = 11  # 1.0s history (11 frames at 10Hz)
MAX_FUTURE_LEN = 80  # 8.0s future (80 frames at 10Hz)


def parse_args():
    parser = argparse.ArgumentParser(description='Animate MTR predictions as GIF showing future N seconds')
    parser.add_argument('--result_pkl', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--scenario_id', type=str, default=None)
    parser.add_argument('--object_index', type=int, default=0,
                        help='Index of the track-to-predict object (default: 0)')
    parser.add_argument('--output_dir', type=str, default='./vis_output')
    parser.add_argument('--fps', type=int, default=10, help='GIF frames per second')
    parser.add_argument('--margin', type=float, default=60.0, help='BEV view margin (meters)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--future_seconds', type=float, default=8.0,
                        help='How many future seconds to animate (max 8.0s, default: 8.0)')
    return parser.parse_args()


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def find_scene_file(data_dir, scenario_id):
    candidate = os.path.join(data_dir, f'{scenario_id}.pkl')
    if os.path.isfile(candidate):
        return candidate
    for fname in os.listdir(data_dir):
        if os.path.splitext(fname)[0] == scenario_id:
            return os.path.join(data_dir, fname)
    if not scenario_id.startswith('enc_'):
        candidate2 = os.path.join(data_dir, f'enc_{scenario_id}.pkl')
        if os.path.isfile(candidate2):
            return candidate2
    return None


def group_polylines_by_type(all_polylines):
    type_col = all_polylines[:, -1].astype(int)
    points_xy = all_polylines[:, :2]
    groups = {k: [] for k in MAP_STYLE}

    def type_to_category(gt):
        if gt in LANE_TYPES: return 'lane'
        if gt in ROAD_LINE_TYPES: return 'road_line'
        if gt in ROAD_EDGE_TYPES: return 'road_edge'
        if gt == CROSSWALK_TYPE: return 'crosswalk'
        return None

    if len(all_polylines) == 0:
        return groups

    cur_type = type_col[0]
    cur_start = 0
    for i in range(1, len(type_col)):
        if type_col[i] != cur_type:
            cat = type_to_category(cur_type)
            if cat is not None:
                groups[cat].append(points_xy[cur_start:i])
            cur_type = type_col[i]
            cur_start = i
    cat = type_to_category(cur_type)
    if cat is not None:
        groups[cat].append(points_xy[cur_start:])
    return groups


def plot_map(ax, groups):
    for cat, segs in groups.items():
        if not segs:
            continue
        style = MAP_STYLE[cat]
        for seg in segs:
            if len(seg) >= 2:
                ax.plot(seg[:, 0], seg[:, 1], '-', color=style['color'],
                        lw=style['lw'], alpha=style['alpha'], zorder=1)


def main():
    args = parse_args()

    # ── Determine number of future frames to animate ─────────────────────
    future_frames = min(int(args.future_seconds / DT), MAX_FUTURE_LEN)
    future_frames = max(future_frames, 1)
    actual_future_seconds = future_frames * DT

    # ── Load result.pkl ──────────────────────────────────────────────────
    with open(args.result_pkl, 'rb') as f:
        all_preds = pickle.load(f)

    # Find the target scenario
    target_sid = args.scenario_id
    found = None
    for scene_preds in all_preds:
        if len(scene_preds) == 0:
            continue
        sid = scene_preds[0]['scenario_id']
        if target_sid is None or sid == target_sid:
            if args.object_index < len(scene_preds):
                found = scene_preds[args.object_index]
                break

    if found is None:
        print(f'ERROR: scenario_id={target_sid}, object_index={args.object_index} not found in result.pkl')
        print(f'Available scenarios: {[scene[0]["scenario_id"] for scene in all_preds if len(scene) > 0][:5]}...')
        return

    scenario_id = found['scenario_id']
    object_id = found['object_id']
    object_type = found['object_type']
    pred_trajs = found['pred_trajs']     # (6, 80, 2)
    pred_scores = found['pred_scores']   # (6,)
    gt_trajs = found['gt_trajs']          # (91, 10)

    print(f'Scenario: {scenario_id}')
    print(f'Object: id={object_id}, type={object_type}')
    print(f'pred_trajs: {pred_trajs.shape}, pred_scores: {pred_scores.shape}')
    print(f'pred_scores: {pred_scores}, sum={pred_scores.sum():.4f}')

    # Check if scores are uniform
    scores_uniform = pred_scores.std() < 0.001
    if scores_uniform:
        print('WARNING: All confidence scores are nearly identical (model undertrained).')

    # ── Load scene data for map ──────────────────────────────────────────
    scene_file = find_scene_file(args.data_dir, scenario_id)
    if scene_file is None:
        print(f'ERROR: scene file not found for {scenario_id} in {args.data_dir}')
        return
    print(f'Loading scene: {scene_file}')

    with open(scene_file, 'rb') as f:
        scene_data = pickle.load(f)

    all_polylines = scene_data['map_infos']['all_polylines']
    current_time_index = scene_data.get('current_time_index', 10)

    # Split GT
    valid_mask = gt_trajs[:, -1] > 0.5
    hist_xy = gt_trajs[:current_time_index + 1, :2]
    future_xy = gt_trajs[current_time_index + 1:, :2]
    future_valid = valid_mask[current_time_index + 1:]

    # ── Setup figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    groups = group_polylines_by_type(all_polylines)

    # Compute view bounds around agent
    center = hist_xy[-1] if len(hist_xy) > 0 else np.zeros(2)
    margin = args.margin
    xlim = (center[0] - margin, center[0] + margin)
    ylim = (center[1] - margin, center[1] + margin)

    # Sort modes by score (best first)
    mode_order = np.argsort(-pred_scores)

    # Distinct colors + line styles for each mode
    mode_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    mode_styles = ['-', '--', '-.', ':', '-', '--']
    mode_widths = [2.5, 1.8, 1.5, 1.2, 1.0, 0.8]

    if scores_uniform:
        # When scores are uniform, use rank-based opacity
        scores_norm = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.15])
    else:
        scores_norm = pred_scores / (pred_scores.max() + 1e-8)

    total_frames = future_frames

    def update(frame):
        ax.clear()

        # Map (static)
        plot_map(ax, groups)

        # History trajectory (blue)
        if len(hist_xy) >= 2:
            ax.plot(hist_xy[:, 0], hist_xy[:, 1], '-', color='blue',
                    linewidth=2.5, zorder=10, label='History')

        # Current agent position at this frame
        if frame == 0:
            cur_pos = hist_xy[-1]
        else:
            best_mode = mode_order[0]
            cur_pos = pred_trajs[best_mode][min(frame - 1, MAX_FUTURE_LEN - 1)]

        ax.scatter(cur_pos[0], cur_pos[1], c='red', s=100, zorder=15,
                   edgecolors='white', linewidths=1.5, marker='*')

        # Predicted trajectories — draw up to current frame
        for rank, mode_idx in enumerate(mode_order):
            traj = pred_trajs[mode_idx]
            n_pts = min(frame + 1, total_frames)
            seg = traj[:n_pts]

            alpha = 0.3 + 0.7 * scores_norm[rank]
            color = mode_colors[rank % len(mode_colors)]
            ls = mode_styles[rank % len(mode_styles)]
            lw = mode_widths[rank % len(mode_widths)]
            label = f'Mode {rank+1} (score={pred_scores[mode_idx]:.4f})'

            ax.plot(seg[:, 0], seg[:, 1], linestyle=ls, color=color, linewidth=lw,
                    alpha=alpha, zorder=6, label=label)

            # Endpoint marker
            ax.scatter(seg[-1, 0], seg[-1, 1], c=[color], s=40,
                       alpha=alpha, zorder=7, edgecolors='white', linewidths=0.5)

        # GT future — draw up to current frame
        if np.any(future_valid):
            n_gt = min(frame + 1, future_xy.shape[0])
            gt_seg = future_xy[:n_gt]
            gt_vis = future_valid[:n_gt]
            if np.any(gt_vis):
                gt_x = gt_seg[gt_vis, 0]
                gt_y = gt_seg[gt_vis, 1]
                ax.plot(gt_x, gt_y, '-', color='black', linewidth=2.5,
                        zorder=12, label='GT future')
                ax.scatter(gt_x[-1], gt_y[-1], c='black', s=40,
                           zorder=13, edgecolors='white', linewidths=0.5)

        # Formatting
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        time_s = frame * DT
        title = f'{scenario_id}\n{object_type} (id={object_id})  t={time_s:.1f}s / {actual_future_seconds:.1f}s'
        if scores_uniform:
            title += '  [WARN: uniform scores - model undertrained]'
        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.2)

    # ── Create animation ─────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_name = f'animation_{scenario_id}_{object_id}_{actual_future_seconds:.0f}s.gif'
    out_path = os.path.join(args.output_dir, out_name)

    print(f'Generating {total_frames} frames ({actual_future_seconds:.1f}s @ {DT*10:.0f}Hz)...')
    interval_ms = int(1000 / args.fps / args.speed)

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=interval_ms, blit=False
    )

    anim.save(out_path, writer='pillow', fps=args.fps)
    plt.close(fig)
    print(f'Saved: {out_path}')
    print(f'Duration: {actual_future_seconds:.1f}s = {total_frames} frames @ {args.fps} fps')


if __name__ == '__main__':
    main()
