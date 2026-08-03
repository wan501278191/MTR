#!/usr/bin/env python3
"""
Waymax-based BEV visualization for MTR trajectory predictions.

Uses waymax's matplotlib backend to render:
  - Roadgraph (lanes, edges, lines, crosswalks) with proper styling
  - All agents as oriented bounding boxes with history trails
  - Predicted trajectories (6 modes, opacity proportional to confidence)
  - Ground-truth future trajectory

Generates both static PNG and animated GIF.

Usage:
    python visualize_waymax.py \
        --result_pkl ../output/waymo/mtr_voyah_smoke/local_smoke/eval/epoch_30/smoke_testA/result.pkl \
        --data_dir ../../data/processed_scenarios_testing_A_full \
        --scenario_id enc_00ab2662a208ecc4a140e9a9 \
        --object_index 0 --output_dir ../output/visualization_waymax \
        --fps 10 --future_seconds 8.0
"""
import _init_path
import argparse
import os
import pickle

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from waymax import datatypes
from waymax import config as waymax_config
from waymax.visualization import utils as viz_utils
from waymax.visualization import viz as waymax_viz

DT = 0.1
MAX_FUTURE = 80

# String type -> waymax int type (0=unset,1=vehicle,2=pedestrian,3=cyclist,4=other)
OBJ_TYPE_MAP = {
    'TYPE_VEHICLE': 1, 'TYPE_PEDESTRIAN': 2, 'TYPE_CYCLIST': 3, 'TYPE_OTHER': 4,
    1: 1, 2: 2, 3: 3, 4: 4,
}

MODE_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']


def parse_args():
    p = argparse.ArgumentParser(description='Waymax-based BEV visualization')
    p.add_argument('--result_pkl', type=str, required=True)
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--scenario_id', type=str, default=None)
    p.add_argument('--object_index', type=int, default=0)
    p.add_argument('--output_dir', type=str, default='./vis_waymax')
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--margin', type=float, default=60.0)
    p.add_argument('--future_seconds', type=float, default=8.0)
    p.add_argument('--top_k', type=int, default=3)
    return p.parse_args()


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
    return None


def build_trajectory(trajs, num_timesteps):
    N = trajs.shape[0]
    return datatypes.Trajectory(
        x=jnp.array(trajs[:, :, 0].astype(np.float32)),
        y=jnp.array(trajs[:, :, 1].astype(np.float32)),
        z=jnp.zeros((N, num_timesteps), dtype=jnp.float32),
        vel_x=jnp.array(trajs[:, :, 7].astype(np.float32)),
        vel_y=jnp.array(trajs[:, :, 8].astype(np.float32)),
        yaw=jnp.array(trajs[:, :, 6].astype(np.float32)),
        valid=jnp.array((trajs[:, :, -1] > 0.5).astype(bool)),
        timestamp_micros=jnp.zeros((N, num_timesteps), dtype=jnp.int32),
        length=jnp.array(trajs[:, :, 3].astype(np.float32)),
        width=jnp.array(trajs[:, :, 4].astype(np.float32)),
        height=jnp.array(trajs[:, :, 5].astype(np.float32)),
    )


def build_object_metadata(track_infos, num_objects, sdc_track_index=0):
    obj_types = track_infos['object_type']
    types_arr = np.zeros(num_objects, dtype=np.int32)
    for i, t in enumerate(obj_types):
        types_arr[i] = OBJ_TYPE_MAP.get(t, OBJ_TYPE_MAP.get(int(t), 4) if isinstance(t, (int, np.integer)) else 4)

    is_sdc_arr = np.zeros(num_objects, dtype=bool)
    if sdc_track_index is not None and 0 <= sdc_track_index < num_objects:
        is_sdc_arr[sdc_track_index] = True

    return datatypes.ObjectMetadata(
        ids=jnp.arange(num_objects, dtype=jnp.int32),
        object_types=jnp.array(types_arr),
        is_sdc=jnp.array(is_sdc_arr),
        is_modeled=jnp.zeros(num_objects, dtype=bool),
        is_valid=jnp.ones(num_objects, dtype=bool),
        objects_of_interest=jnp.zeros(num_objects, dtype=bool),
        is_controlled=jnp.zeros(num_objects, dtype=bool),
    )


def build_roadgraph(map_infos):
    poly = map_infos['all_polylines']
    N = poly.shape[0]
    return datatypes.RoadgraphPoints(
        x=jnp.array(poly[:, 0].astype(np.float32)),
        y=jnp.array(poly[:, 1].astype(np.float32)),
        z=jnp.array(poly[:, 2].astype(np.float32) if poly.shape[1] > 2 else np.zeros(N, dtype=np.float32)),
        dir_x=jnp.array(poly[:, 3].astype(np.float32) if poly.shape[1] > 3 else np.zeros(N, dtype=np.float32)),
        dir_y=jnp.array(poly[:, 4].astype(np.float32) if poly.shape[1] > 4 else np.zeros(N, dtype=np.float32)),
        dir_z=jnp.zeros(N, dtype=jnp.float32),
        types=jnp.array(poly[:, -1].astype(np.int32)),
        ids=jnp.arange(N, dtype=jnp.int32),
        valid=jnp.ones(N, dtype=bool),
    )


def build_simulator_state(scene_data, current_time_index):
    track_infos = scene_data['track_infos']
    trajs = track_infos['trajs']
    N, T, _ = trajs.shape
    traj = build_trajectory(trajs, T)
    sdc_idx = scene_data.get('sdc_track_index', 0)
    metadata = build_object_metadata(track_infos, N, sdc_idx)
    roadgraph = build_roadgraph(scene_data['map_infos'])
    traffic_lights = datatypes.TrafficLights(
        x=jnp.zeros((0, T), dtype=jnp.float32),
        y=jnp.zeros((0, T), dtype=jnp.float32),
        z=jnp.zeros((0, T), dtype=jnp.float32),
        state=jnp.zeros((0, T), dtype=jnp.int32),
        lane_ids=jnp.zeros((0, T), dtype=jnp.int32),
        valid=jnp.zeros((0, T), dtype=bool),
    )
    return datatypes.SimulatorState(
        sim_trajectory=traj,
        log_trajectory=traj,
        log_traffic_light=traffic_lights,
        object_metadata=metadata,
        timestep=int(current_time_index),
        roadgraph_points=roadgraph,
    )


def plot_predictions_overlay(ax, pred_trajs, pred_scores, gt_future_xy,
                             frame, top_k=3, future_frames=80):
    mode_order = np.argsort(-pred_scores)
    scores_norm = pred_scores / (pred_scores.max() + 1e-8)
    k = min(top_k, len(mode_order))
    for rank in range(k):
        mode_idx = mode_order[rank]
        traj = pred_trajs[mode_idx]
        n_pts = min(frame + 1, future_frames)
        seg = traj[:n_pts]
        alpha = 0.3 + 0.7 * scores_norm[mode_idx]
        color = MODE_COLORS[rank % len(MODE_COLORS)]
        lw = 3.0 if rank == 0 else 1.5
        ax.plot(seg[:, 0], seg[:, 1], '-', color=color, linewidth=lw,
                alpha=alpha, zorder=20)
        ax.scatter(seg[-1, 0], seg[-1, 1], c=[color], s=50,
                   alpha=alpha, zorder=21, edgecolors='white', linewidths=0.5)
    n_gt = min(frame + 1, gt_future_xy.shape[0])
    if n_gt > 0:
        gt_seg = gt_future_xy[:n_gt]
        ax.plot(gt_seg[:, 0], gt_seg[:, 1], '-', color='black', linewidth=2.5, zorder=18)
        ax.scatter(gt_seg[-1, 0], gt_seg[-1, 1], c='black', s=50, zorder=19,
                   edgecolors='white', linewidths=0.5)


def main():
    args = parse_args()
    future_frames = min(int(args.future_seconds / DT), MAX_FUTURE)

    results = load_pkl(args.result_pkl)
    scenario_id = args.scenario_id or results[0][0]['scenario_id']
    scene_preds = None
    for scene in results:
        if scene[0]['scenario_id'] == scenario_id:
            scene_preds = scene
            break
    if scene_preds is None:
        scene_preds = results[0]
        scenario_id = scene_preds[0]['scenario_id']

    pred = scene_preds[args.object_index]
    pred_trajs = pred['pred_trajs']
    pred_scores = pred['pred_scores']
    object_id = pred['object_id']
    object_type = pred.get('object_type', 'unknown')
    gt_trajs = pred['gt_trajs']

    scene_file = find_scene_file(args.data_dir, scenario_id)
    if not scene_file:
        print(f'Scene file not found for {scenario_id} in {args.data_dir}')
        return
    with open(scene_file, 'rb') as f:
        scene_data = pickle.load(f)

    current_time_index = scene_data.get('current_time_index', 10)
    state = build_simulator_state(scene_data, current_time_index)
    gt_future_xy = gt_trajs[current_time_index + 1:, :2]
    hist_xy = gt_trajs[:current_time_index + 1, :2]
    track_index = pred.get('track_index_to_predict', args.object_index)
    if track_index >= state.object_metadata.num_objects:
        track_index = args.object_index

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Static PNG ──────────────────────────────────────────────────
    print('Generating static PNG...')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    viz_config = viz_utils.VizConfig(
        front_x=args.margin, back_x=args.margin,
        front_y=args.margin, back_y=args.margin,
        px_per_meter=4.0, show_agent_id=True,
        center_agent_idx=-1,
    )
    waymax_viz.plot_simulator_state_matplotlib(
        ax, state, viz_config, use_log_traj=True,
        highlight_obj=waymax_config.ObjectType.SDC,
    )
    # Manually center on predicted agent (waymax center_agent_idx has a bug)
    agent_xy = hist_xy[-1] if len(hist_xy) > 0 else np.zeros(2)
    ax.set_xlim(agent_xy[0] - args.margin, agent_xy[0] + args.margin)
    ax.set_ylim(agent_xy[1] - args.margin, agent_xy[1] + args.margin)
    # Overlay all prediction modes
    mode_order = np.argsort(-pred_scores)
    scores_norm = pred_scores / (pred_scores.max() + 1e-8)
    for rank, mode_idx in enumerate(mode_order):
        traj = pred_trajs[mode_idx]
        alpha = 0.2 + 0.8 * scores_norm[mode_idx]
        color = MODE_COLORS[rank % len(MODE_COLORS)]
        lw = 3.0 if rank == 0 else 1.2
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=lw,
                alpha=alpha, zorder=20,
                label=f'Mode {rank+1} (score={pred_scores[mode_idx]:.4f})')
        ax.scatter(traj[-1, 0], traj[-1, 1], c=[color], s=80,
                   alpha=alpha, zorder=21, edgecolors='white', linewidths=1)
    gt_valid = gt_trajs[current_time_index + 1:, -1] > 0.5
    gt_fx = gt_future_xy[gt_valid]
    if len(gt_fx) > 0:
        ax.plot(gt_fx[:, 0], gt_fx[:, 1], '-', color='black', linewidth=3,
                zorder=18, label='Ground Truth')
        ax.scatter(gt_fx[-1, 0], gt_fx[-1, 1], c='black', s=80,
                   zorder=19, edgecolors='white', linewidths=1)
    ax.set_aspect('equal')
    ax.set_title(f'Waymax BEV — {scenario_id}\n{object_type} (id={object_id})', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    png_path = os.path.join(args.output_dir, f'waymax_static_{scenario_id}_{object_id}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {png_path}')

    # ── 2. Animated GIF ────────────────────────────────────────────────
    print(f'Generating animation ({future_frames} frames, {args.future_seconds:.1f}s)...')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    def update(frame):
        ax.clear()
        ts = min(current_time_index + frame, state.log_trajectory.num_timesteps - 1)
        state_t = datatypes.SimulatorState(
            sim_trajectory=state.sim_trajectory,
            log_trajectory=state.log_trajectory,
            log_traffic_light=state.log_traffic_light,
            object_metadata=state.object_metadata,
            timestep=int(ts),
            roadgraph_points=state.roadgraph_points,
        )
        waymax_viz.plot_simulator_state_matplotlib(
            ax, state_t, viz_config, use_log_traj=True,
        )
        # Manually center on predicted agent
        agent_xy = hist_xy[-1] if len(hist_xy) > 0 else np.zeros(2)
        ax.set_xlim(agent_xy[0] - args.margin, agent_xy[0] + args.margin)
        ax.set_ylim(agent_xy[1] - args.margin, agent_xy[1] + args.margin)
        plot_predictions_overlay(ax, pred_trajs, pred_scores, gt_future_xy,
                                 frame, top_k=args.top_k, future_frames=future_frames)
        time_s = frame * DT
        ax.set_title(f'Waymax BEV — {scenario_id}\n{object_type} (id={object_id})  t={time_s:.1f}s/{args.future_seconds:.1f}s',
                     fontsize=11)
        ax.grid(True, alpha=0.15)
    interval_ms = int(1000 / args.fps)
    anim = animation.FuncAnimation(fig, update, frames=future_frames,
                                   interval=interval_ms, blit=False)
    gif_path = os.path.join(args.output_dir, f'waymax_anim_{scenario_id}_{object_id}_{args.future_seconds:.0f}s.gif')
    anim.save(gif_path, writer='pillow', fps=args.fps)
    plt.close(fig)
    print(f'Saved: {gif_path}')

    print(f'\n=== Prediction Scores ===')
    for i, (idx, score) in enumerate(zip(np.argsort(-pred_scores), sorted(pred_scores, reverse=True))):
        print(f'  Mode {i+1}: score={score:.6f}')


if __name__ == '__main__':
    main()
