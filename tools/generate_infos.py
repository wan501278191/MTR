#!/usr/bin/env python3
"""
Generate infos pkl files from preprocessed scenario pkl files.
Extracts metadata subset needed by WaymoDataset from each enc_*.pkl scene file.

Usage:
    python generate_infos.py --data_dir ../data/processed_scenarios_validation --output ../data/processed_scenarios_val_infos.pkl
    python generate_infos.py --all  # Generate all splits at once
"""
import os
import sys
import argparse
import pickle
from glob import glob
from tqdm import tqdm

# Info keys to extract (subset of scene data, no track_infos/map_infos/dynamic_map_infos)
INFO_KEYS = [
    'scenario_id',
    'timestamps_seconds',
    'current_time_index',
    'sdc_track_index',
    'objects_of_interest',
    'tracks_to_predict',
]

# Split definitions: (split_name, data_dir, output_file)
SPLITS = [
    ('training', 'processed_scenarios_training', 'processed_scenarios_training_infos.pkl'),
    ('validation', 'processed_scenarios_validation', 'processed_scenarios_val_infos.pkl'),
    ('testA_part', 'processed_scenarios_testing_A_part', 'processed_scenarios_testA_part_infos.pkl'),
    ('testA_full', 'processed_scenarios_testing_A_full', 'processed_scenarios_testA_full_infos.pkl'),
    ('testB1_part', 'processed_scenarios_testing_B1_part', 'processed_scenarios_testB1_part_infos.pkl'),
]


def generate_infos(data_dir, output_file):
    """Generate infos pkl from a directory of enc_*.pkl scene files."""
    scene_files = sorted(glob(os.path.join(data_dir, 'enc_*.pkl')))
    print(f"Found {len(scene_files)} scene files in {data_dir}")

    infos = []
    for scene_file in tqdm(scene_files, desc=f"Generating {os.path.basename(output_file)}"):
        with open(scene_file, 'rb') as f:
            scene_data = pickle.load(f)

        # Extract metadata subset
        info = {key: scene_data[key] for key in INFO_KEYS if key in scene_data}
        infos.append(info)

    # Save infos pkl
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(infos, f)
    print(f"✓ Saved {len(infos)} infos to {output_file}")
    return infos


def verify_infos(infos, data_dir):
    """Verify generated infos against source files."""
    scene_files = glob(os.path.join(data_dir, 'enc_*.pkl'))
    assert len(infos) == len(scene_files), \
        f"Count mismatch: {len(infos)} infos vs {len(scene_files)} files"

    valid_types = {'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST'}
    for i, info in enumerate(infos):
        assert 'scenario_id' in info, f"Missing scenario_id at index {i}"
        assert info['scenario_id'].startswith('enc_'), \
            f"scenario_id must retain enc_ prefix: {info['scenario_id']}"
        assert 'tracks_to_predict' in info, f"Missing tracks_to_predict at index {i}"
        ttp = info['tracks_to_predict']
        assert len(ttp.keys()) == 3, \
            f"tracks_to_predict must have 3 keys: {ttp.keys()}"
        for ot in ttp.get('object_type', []):
            assert ot in valid_types, f"Unexpected object_type: {ot}"

    print(f"✓ Verification passed: {len(infos)} infos, all invariants hold")


def main():
    parser = argparse.ArgumentParser(description='Generate infos pkl from preprocessed scenario files')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing enc_*.pkl files')
    parser.add_argument('--output', type=str, default=None, help='Output infos pkl path')
    parser.add_argument('--data_root', type=str, default='../data', help='Root data directory (for --all mode)')
    parser.add_argument('--all', action='store_true', help='Generate infos for all splits')
    args = parser.parse_args()

    if args.all:
        for split_name, split_dir, output_name in SPLITS:
            data_dir = os.path.join(args.data_root, split_dir)
            output_file = os.path.join(args.data_root, output_name)
            if not os.path.isdir(data_dir):
                print(f"⚠ Skipping {split_name}: {data_dir} not found")
                continue
            infos = generate_infos(data_dir, output_file)
            verify_infos(infos, data_dir)
        print("\n✅ All splits processed!")
    elif args.data_dir and args.output:
        infos = generate_infos(args.data_dir, args.output)
        verify_infos(infos, args.data_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
