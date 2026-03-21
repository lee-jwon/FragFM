import os
from glob import glob

import lmdb


def merge_lmdbs(db_paths, output_db_path):
    """
    Merge multiple LMDB databases into a single output database.

    Parameters:
    - db_paths (list of str): List of paths to the LMDB databases to be merged.
    - output_db_path (str): Path to the output LMDB database.
    """
    with lmdb.open(output_db_path, map_size=int(3e9 * 20)) as output_env:
        with output_env.begin(write=True) as output_txn:
            for db_path in db_paths:
                with lmdb.open(db_path, readonly=True) as env:
                    with env.begin() as txn:
                        for key, value in txn.cursor():
                            output_txn.put(key, value)
    print(f"[LOG] Merged {len(db_paths)} LMDB databases into {output_db_path}")


import shutil
import sys

dirn = sys.argv[1]
lmdb_dirns = glob(os.path.join(dirn, "*_*.lmdb"))

tag = os.path.basename(os.path.normpath(dirn))

print(len(lmdb_dirns))
merge_lmdbs(lmdb_dirns, f"./data/processed/{tag}.lmdb")

# for db_path in lmdb_dirns:
#    shutil.rmtree(db_path)
# print(f"[LOG] Removed {len(lmdb_dirns)} shard LMDB directories")
