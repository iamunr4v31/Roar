"""
This script computes features for TTS models prior to training, such as pitch and energy.
The resulting features will be stored in the provided 'feature_dir'.

$ python <roar_root_path>/scripts/dataset_processing/tts/compute_features.py \
    --feature_config_path=<roar_root_path>/examples/tts/conf/features/feature_22050.yaml \
    --manifest_path=<data_root_path>/manifest.json \
    --audio_dir=<data_root_path>/audio \
    --feature_dir=<data_root_path>/features \
    --num_workers=1
"""

import argparse
from pathlib import Path

from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm

from roar.collections.asr.parts.utils.manifest_utils import read_manifest


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute TTS features.",
    )
    parser.add_argument(
        "--feature_config_path",
        required=True,
        type=Path,
        help="Path to feature config file.",
    )
    parser.add_argument(
        "--manifest_path",
        required=True,
        type=Path,
        help="Path to training manifest.",
    )
    parser.add_argument(
        "--audio_dir",
        required=True,
        type=Path,
        help="Path to base directory with audio data.",
    )
    parser.add_argument(
        "--feature_dir",
        required=True,
        type=Path,
        help="Path to directory where feature data will be stored.",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of parallel threads to use. If -1 all CPUs are used.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    feature_config_path = args.feature_config_path
    manifest_path = args.manifest_path
    audio_dir = args.audio_dir
    feature_dir = args.feature_dir
    num_workers = args.num_workers

    if not manifest_path.exists():
        raise ValueError(f"Manifest {manifest_path} does not exist.")

    if not audio_dir.exists():
        raise ValueError(f"Audio directory {audio_dir} does not exist.")

    feature_config = OmegaConf.load(feature_config_path)
    feature_config = instantiate(feature_config)
    featurizers = feature_config.featurizers

    entries = read_manifest(manifest_path)

    for feature_name, featurizer in featurizers.items():
        print(f"Computing: {feature_name}")
        Parallel(n_jobs=num_workers)(
            delayed(featurizer.save)(
                manifest_entry=entry,
                audio_dir=audio_dir,
                feature_dir=feature_dir,
            )
            for entry in tqdm(entries)
        )


if __name__ == "__main__":
    main()
