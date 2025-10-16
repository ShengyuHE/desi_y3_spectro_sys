#!/usr/bin/env python3
import os
import shutil
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)

DESI_ROOT = os.environ.get("DESI_ROOT", "/global/cfs/cdirs/desi")
SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/s/shengyu")

for mocknum in range(25):  # 0 to 24
    directory = f"Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{mocknum}/mock{mocknum}/LSScats"
    source_dir = Path(DESI_ROOT) / "survey/catalogs" / directory
    target_dir = Path(SCRATCH) / "galaxies/catalogs" / directory

    if not source_dir.exists():
        logging.warning(f"Source directory missing: {source_dir}")
        continue
    # Make target directory if it does not exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over QSO files
    for file_path in source_dir.glob("QSO*"):
        fname = file_path.name
        # Skip HPmapcut files
        if "HPmapcut" in fname:
            continue

        # if any(x in fname for x in ["15", "16", "17"]):
            # continue

        # Skip NGC and SGC files
        # if "NGC" in fname or "SGC" in fname:
            # continue

        # Copy if it does not exist
        target_file = target_dir / fname
        if target_file.exists():
            logging.info(f"Skipping {fname} (already exists in {target_dir})")
        else:
            logging.info(f"Copying {file_path} -> {target_file}")
            shutil.copy2(file_path, target_file)

        subdirs = ["dv_obs_z0.8-2.1", "dv_obs_z0.8-2.1_evol0.1"]
        for sd in subdirs:
            sub_target_dir = target_dir / sd
            sub_target_dir.mkdir(parents=True, exist_ok=True) 
            target_file = sub_target_dir / fname
            if target_file.exists():
                logging.info(f"Skipping {fname} (already exists in {sub_target_dir})")
            else:
                logging.info(f"Copying {file_path} -> {target_file}")
                shutil.copy2(file_path, target_file)