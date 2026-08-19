#!/usr/bin/env python3
"""
Visualize MTR trajectory predictions in BEV (bird's eye view).

Loads result.pkl (prediction output from test.py) and the corresponding
scene pkl (enc_*.pkl) to produce a matplotlib BEV plot showing:
  - Map polylines (lane=green, road_edge=red, crosswalk=gray, road_line=yellow)
  - Historical trajectory (blue, frames 0..current_time_index)
  - Predicted trajectories (6 modes, opacity proportional to score)
  - Ground-truth future trajectory (black solid line)

Usage:
    python visualize_prediction.py \
        --result_pkl /path/to/result.pkl \
        --data_dir /path/to/processed_scenarios_validation \
        --scenario_id enc_xxxx \
        --output_dir ./vis_out
"""
import _init_path
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for PNG output
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ── polyline global_type → category mapping (from waymo_types.py) ──────────
LANE_TYPES = {1, 2, 3}                        # freeway, surface_street, bike_lane
ROAD_LINE_TYPES = {6, 7, 8, 9, 10, 11, 12, 13}
ROAD_EDGE_TYPES = {15, 16}                    # boundary, median
CROSSWALK_TYPE = 18
STOP_SIGN_TYPE = 17
SPEED_BUMP_TYPE = 19

MAP_STYLE = {
    'lane':       {'color': 'green',  'lw': 0.5, 'alpha': 0.5},
    'road_line':  {'color': 'yellow', 'lw': 0.5, 'alpha': 0.6},
    'road_edge':  {'color': 'red',    'lw': 0.8, 'alpha': 0.6},
    'crosswalk':  {'color': 'gray',   'lw': 0.5, 'alpha': 0.3},
    'stop_sign':  {'color': 'red',    'lw': 0.5, 'alpha': 0.4, 'marker': 'x'},
    'speed_bump': {'color': 'gray',   'lw': 0.5, 'alpha': 0.3, 'marker': 's'},
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MTR predictions in BEV')
    parser.add_argument('--result_pkl', type=str, required=True,
                        help='Path to result.pkl from test.py')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing enc_*.pkl scene files')
    parser.add_argument('--scenario_id', type=str, default=None,
                        help='Scenario ID to visualize (default: first in result)')
    parser.add_argument('--object_index', type=int, default=0,
                        help='Index of the track-to-predict object within the '
                             'scenario (default: 0 = first)')
    parser.add_argument('--output_dir', type=str, default='./vis_output',
                        help='Output directory for PNG files')
    parser.add_argument('--all', action='store_true', default=False,
                        help='Visualize all scenarios in result.pkl')
    parser.add_argument('--max_scenarios', type=int, default=3,
                        help='Max number of scenarios to visualize (default: 3)')
    return parser.parse_args()


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def find_scene_file(data_dir, scenario_id):
    """Find the scene pkl for *scenario_id* inside *data_dir*.

    The scenario_id stored in the data already carries the ``enc_`` prefix,
    so the file name is simply ``{scenario_id}.pkl``.  We fall back to a
    glob search just in case the naming convention differs.
    """
    # Primary path: scenario_id.pkl (scenario_id already starts with enc_)
    candidate = os.path.join(data_dir, f'{scenario_id}.pkl')
    if os.path.isfile(candidate):
        return candidate

    # Fallback: search any file whose stem matches
    for fname in os.listdir(data_dir):
        stem = os.path.splitext(fname)[0]
        if stem == scenario_id:
            return os.path.join(data_dir, fname)

    # Last resort: try with enc_ prefix if it was stripped
    if not scenario_id.startswith('enc_'):
        candidate2 = os.path.join(data_dir, f'enc_{scenario_id}.pkl')
        if os.path.isfile(candidate2):
            return candidate2

    return None


def group_polylines_by_type(all_polylines):
    """Group points of *all_polylines* (N, 7) by their global_type (last col).

    Returns a dict mapping category name → list of (M, 2) arrays of [x, y].
    Consecutive points of the same type are treated as one polyline segment
    list.
    """
    type_col = all_polylines[:, -1].astype(int)
    points_xy = all_polylines[:, :2]

    groups = {k: [] for k in MAP_STYLE}
    # Map global_type int → category
    def type_to_category(gt):
        if gt in LANE_TYPES:
            return 'lane'
        if gt in ROAD_LINE_TYPES:
            return 'road_line'
        if gt in ROAD_EDGE_TYPES:
            return 'road_edge'
        if gt == CROSSWALK_TYPE:
            return 'crosswalk'
        if gt == STOP_SIGN_TYPE:
            return 'stop_sign'
        if gt == SPEED_BUMP_TYPE:
            return 'speed_bump'
        return None

    # Split into contiguous runs of the same type
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
    # final run
    cat = type_to_category(cur_type)
    if cat is not None:
        groups[cat].append(points_xy[cur_start:])

    return groups


def plot_map(ax, groups):
    """Draw all map polylines on *ax* using the style table."""
    for cat, segs in groups.items():
        style = MAP_STYLE[cat]
        if cat in ('stop_sign', 'speed_bump'):
            # Points, not lines
            for seg in segs:
                if len(seg) > 0:
                    ax.scatter(seg[:, 0], seg[:, 1],
                               c=style['color'], s=20,
                               alpha=style['alpha'], marker=style['marker'],
                               zorder=1)
        else:
            segs_xy = [seg for seg in segs if len(seg) >= 2]
            if segs_xy:
                # Build LineCollection segments
                segments = []
                for seg in segs_xy:
                    segments.append(np.stack([seg[:-1], seg[1:]], axis=1))
                if segments:
                    all_segs = np.concatenate(segments, axis=0)
                    lc = LineCollection(all_segs, colors=style['color'],
                                        linewidths=style['lw'],
                                        alpha=style['alpha'], zorder=1)
                    ax.add_collection(lc)


def add_arrows(ax, traj_xy, color, step=10, scale=1.0, alpha=0.8):
    """Draw directional arrows along a trajectory every *step* points."""
    for i in range(step, len(traj_xy), step):
        dx = traj_xy[i, 0] - traj_xy[i - step, 0]
        dy = traj_xy[i, 1] - traj_xy[i - step, 1]
        if dx == 0 and dy == 0:
            continue
        ax.annotate('', xy=(traj_xy[i, 0], traj_xy[i, 1]),
                    xytext=(traj_xy[i - step, 0], traj_xy[i - step, 1]),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.2, alpha=alpha),
                    zorder=5)


def visualize_single(args, result_pkl_path, data_dir, output_dir, scenario_id, object_index):
    """Visualize a single scenario + object."""
    results = load_pkl(result_pkl_path)

    scene_preds = {}
    for scene_list in results:
        for pred_dict in scene_list:
            sid = pred_dict['scenario_id']
            scene_preds.setdefault(sid, []).append(pred_dict)

    if scenario_id not in scene_preds:
        raise KeyError(f'scenario_id {scenario_id!r} not found in result_pkl. '
                       f'Available: {sorted(scene_preds.keys())[:10]} ...')

    pred_list = scene_preds[scenario_id]
    if object_index >= len(pred_list):
        raise IndexError(f'object_index {object_index} out of range '
                         f'({len(pred_list)} objects in scenario)')
    pred = pred_list[object_index]

    scene_path = find_scene_file(data_dir, scenario_id)
    if scene_path is None:
        raise FileNotFoundError(
            f'Scene pkl for scenario_id={scenario_id!r} not found in '
            f'{data_dir}')
    scene = load_pkl(scene_path)

    current_time_index = scene['current_time_index']
    map_infos = scene['map_infos']
    all_polylines = map_infos['all_polylines']

    pred_trajs = np.asarray(pred['pred_trajs'])
    pred_scores = np.asarray(pred['pred_scores'])
    gt_trajs = np.asarray(pred['gt_trajs'])
    object_id = pred['object_id']
    object_type = pred['object_type']

    valid_mask = gt_trajs[:, -1] > 0.5
    hist_xy = gt_trajs[:current_time_index + 1, :2]
    future_xy = gt_trajs[current_time_index + 1:, :2]
    future_valid = valid_mask[current_time_index + 1:]

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    groups = group_polylines_by_type(all_polylines)
    plot_map(ax, groups)

    if len(hist_xy) >= 2:
        ax.plot(hist_xy[:, 0], hist_xy[:, 1], '-', color='blue',
                linewidth=2.0, zorder=10, label='History')
        ax.scatter(hist_xy[-1, 0], hist_xy[-1, 1], c='blue', s=50,
                   zorder=11, edgecolors='white', linewidths=0.5)

    scores_norm = pred_scores / (pred_scores.max() + 1e-8)
    mode_order = np.argsort(-pred_scores)

    cmap = plt.cm.viridis
    for rank, mode_idx in enumerate(mode_order):
        traj = pred_trajs[mode_idx]
        score = pred_scores[mode_idx]
        alpha = 0.3 + 0.7 * scores_norm[mode_idx]
        color = cmap(rank / max(len(mode_order) - 1, 1))
        label = f'Mode {mode_idx} (score={score:.3f})'
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.5,
                alpha=alpha, zorder=6, label=label)
        add_arrows(ax, traj, color=color, step=15, alpha=alpha)
        ax.scatter(traj[-1, 0], traj[-1, 1], c=[color], s=30,
                   alpha=alpha, zorder=7, edgecolors='white', linewidths=0.3)

    if np.any(future_valid):
        future_x = future_xy[future_valid, 0]
        future_y = future_xy[future_valid, 1]
        ax.plot(future_x, future_y, '-', color='black', linewidth=2.0,
                zorder=12, label='GT future')
        ax.scatter(future_x[-1], future_y[-1], c='black', s=30,
                   zorder=13, edgecolors='white', linewidths=0.3)

    ax.set_aspect('equal')
    ax.set_xlabel('X (world)')
    ax.set_ylabel('Y (world)')
    ax.set_title(f'{scenario_id} | {object_type} (id={object_id})')

    center = hist_xy[-1] if len(hist_xy) > 0 else np.zeros(2)
    margin = 60.0
    ax.set_xlim(center[0] - margin, center[0] + margin)
    ax.set_ylim(center[1] - margin, center[1] + margin)

    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_name = f'visualization_{scenario_id}_{object_id}.png'
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    args = parse_args()

    if not os.path.isfile(args.result_pkl):
        raise FileNotFoundError(f'result_pkl not found: {args.result_pkl}')
    results = load_pkl(args.result_pkl)

    scene_preds = {}
    for scene_list in results:
        for pred_dict in scene_list:
            sid = pred_dict['scenario_id']
            scene_preds.setdefault(sid, []).append(pred_dict)

    if args.all:
        scenario_ids = sorted(scene_preds.keys())
        if args.max_scenarios > 0:
            scenario_ids = scenario_ids[:args.max_scenarios]
        print(f'Visualizing {len(scenario_ids)} scenarios...')
        for i, sid in enumerate(scenario_ids):
            n_objects = len(scene_preds[sid])
            for obj_idx in range(n_objects):
                try:
                    visualize_single(args, args.result_pkl, args.data_dir,
                                     args.output_dir, sid, obj_idx)
                except Exception as e:
                    print(f'  SKIP {sid} obj={obj_idx}: {e}')
            if (i + 1) % 50 == 0:
                print(f'  Progress: {i+1}/{len(scenario_ids)}')
        print(f'Done: {len(scenario_ids)} scenarios visualized')
    else:
        scenario_id = args.scenario_id
        if scenario_id is None:
            scenario_id = sorted(scene_preds.keys())[0]
            print(f'No --scenario_id given, using first: {scenario_id}')
        visualize_single(args, args.result_pkl, args.data_dir,
                         args.output_dir, scenario_id, args.object_index)


if __name__ == '__main__':
    main()
