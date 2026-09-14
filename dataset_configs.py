DATASET_CONFIGS = {
    "vorts0001": {
        "training_mode": "full",
        "path": "./data/xueliao_data/vorts0001.raw",
        "x_voxel_size": 128,
        "y_voxel_size": 128,
        "z_voxel_size": 128,
        "use_tif": False,
        "transpose": True,
        "dtype": "<f4",
        "intensity_range": (-1.0, 1.0),
    },
    "aneurysm": {
        "training_mode": "full",
        "path": "./data/aneurism_256x256x256_uint8.tif",
        "x_voxel_size": 256,
        "y_voxel_size": 256,
        "z_voxel_size": 256,
        "use_tif": True,
        "transpose": False,
    },
    "breast": {
        "training_mode": "full",
        "path": "./data/public/breast_448x448x160_uint8.tif",
        "x_voxel_size": 448,
        "y_voxel_size": 448,
        "z_voxel_size": 160,
        "chunk_size": (448, 448, 160),
        "chunk_len": 2.0,
        "use_tif": True,
        "transpose": True,
    },
    "kistfull": {
        "training_mode": "full",
        "path": '/home/Alexandrite/yjkim/repos/gaussian-splatting-smin/data/vis/kistfull_2537x1295x133_uint16.tif',
        "x_voxel_size": 2537,
        "y_voxel_size": 1295,
        "z_voxel_size": 133,
        "chunk_size": (2537, 1295, 133),
        "chunk_len": 2.0,
        "use_tif": True,
        "transpose": True
    },
    "bigbrain": {
        "training_mode": "chunk_sequential",
        "path": '/home/Alexandrite/yjkim/repos/gaussian-splatting-smin/data/vis/bigbrain_mip2_filtered_1749x1643x1428.raw',
        "x_voxel_size": 1749,
        "y_voxel_size": 1643,
        "z_voxel_size": 1428,
        "use_tif": False,
        "chunk_size": (875, 822, 714),
        "chunk_len": 1.0,
        "transpose": True
    },
    "woodbranch": {
        "training_mode": "chunk_sequential",
        "path": "/home/Alexandrite/yjkim/repos/gaussian-splatting-smin/data/vis/woodbranch_2048x2048x2048_uint16.raw",
        "x_voxel_size": 2048,
        "y_voxel_size": 2048,
        "z_voxel_size": 2048,
        "chunk_size": (1024, 1024, 1024),
        "chunk_len": 1.0,
        "use_tif": False,
        "transpose": True,
    },
    "hemibrain_vDeltaA": {
        "training_mode": "chunk_parallel",
        "path": "/home/Alexandrite/yjkim/repos/gaussian-splatting-smin/data/vis/hemibrain_vDeltaA_n85_mip2_4072x4372x4171.raw",
        "x_voxel_size": 4372, # xyz for training
        "y_voxel_size": 4072,
        "z_voxel_size": 4171,
        "raw_shape": (4171, 4372, 4072), # need xy swap
        "axis_order": (1,2,0),
        "chunk_size": (1093, 1018, 1043),
        "chunk_len": 0.5,
        "use_tif": False,
        "transpose": True,
    },
    "hemibrain_lLN2P": {
        "training_mode": "chunk_parallel",
        "path": "/home/Alexandrite/yjkim/repos/gaussian-splatting-smin/data/vis/hemibrain_lLN2P_top2_n2_mip1_5115x4774x6128.raw",
        "x_voxel_size": 6128,
        "y_voxel_size": 4774,
        "z_voxel_size": 5115,
        "raw_shape": (6128, 4774, 5115),
        "chunk_size": (1226, 955, 1023),
        "chunk_len": 0.4,
        "use_tif": False,
        "transpose": False,
    }
}

# Per-frame entries for the 90-frame time-varying vorts sequence, same grid/
# dtype/range as "vorts0001" above (byte-identical source file, confirmed via
# sha256). Used to run the original train_chunk.py ESVR pipeline (adaptive
# densification, per-parameter LR schedule) on each frame independently, on
# this copy's CUDA-fixed voxelizer, with position_lr rescaled to compensate
# (see output/experiments/vorts0001_cuda_fixed_lr_only/RESULTS.md for the
# single-frame diagnostic this 90-frame run is based on).
for _i in range(1, 91):
    DATASET_CONFIGS[f"vorts_frame_{_i:04d}"] = {
        "training_mode": "full",
        "path": f"./data/xueliao_timevaryingdata/vorts_timevarying/vorts{_i:04d}.raw",
        "x_voxel_size": 128,
        "y_voxel_size": 128,
        "z_voxel_size": 128,
        "use_tif": False,
        "transpose": True,
        "dtype": "<f4",
        "intensity_range": (-1.0, 1.0),
    }
del _i
