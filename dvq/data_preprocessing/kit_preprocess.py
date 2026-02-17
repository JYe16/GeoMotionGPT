"""
KIT Motion-Language Dataset Preprocessing (Vectorized)

Converts raw KIT-ML data (MMM XML) into 251-dim features matching HumanML3D,
using fully vectorized numpy/torch operations for speed.

Feature breakdown (251 dims, 21 joints):
    root_vel_y(1) + root_vel_xz(2) + root_y(1)
    + local_pos(20*3=60) + cont6d(20*6=120)
    + joint_vel(21*3=63) + foot_contact(4)

Usage:
    python dvq/data_preprocessing/kit_preprocess.py \\
        --raw_dir /data/jackieye/KIT-ML/raw/ \\
        --output_dir datasets/kit/
"""

import argparse, json, os, sys
import xml.etree.cElementTree as ET
from typing import List
import numpy as np
import torch
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from motGPT.data.humanml.common.skeleton import Skeleton
from motGPT.data.humanml.common.quaternion import (
    qfix, qbetween_np, qrot_np, qmul_np, qinv_np,
    quaternion_to_cont6d_np,
)
from motGPT.data.humanml.utils.paramUtil import (
    kit_raw_offsets, kit_kinematic_chain, kit_tgt_skel_id,
)


# ============================================================================
# Vectorized rotation utilities (pure numpy, batch over T frames)
# ============================================================================

def _batch_euler_to_rotmat(angles):
    """(T, 3) Euler XYZ -> (T, 3, 3) rotation matrices.  Vectorized."""
    cx, sx = np.cos(angles[:, 0]), np.sin(angles[:, 0])
    cy, sy = np.cos(angles[:, 1]), np.sin(angles[:, 1])
    cz, sz = np.cos(angles[:, 2]), np.sin(angles[:, 2])
    T = len(angles)
    R = np.zeros((T, 3, 3), dtype=np.float64)
    R[:, 0, 0] = cy*cz
    R[:, 0, 1] = sx*sy*cz - cx*sz
    R[:, 0, 2] = cx*sy*cz + sx*sz
    R[:, 1, 0] = cy*sz
    R[:, 1, 1] = sx*sy*sz + cx*cz
    R[:, 1, 2] = cx*sy*sz - sx*cz
    R[:, 2, 0] = -sy
    R[:, 2, 1] = sx*cy
    R[:, 2, 2] = cx*cy
    return R


def _batch_axis_rotmat(angles, axis):
    """(T,) angles + axis char -> (T, 3, 3). Vectorized single-axis rotation."""
    T = len(angles)
    c, s = np.cos(angles), np.sin(angles)
    R = np.zeros((T, 3, 3), dtype=np.float64)
    if axis == 'x':
        R[:, 0, 0] = 1; R[:, 1, 1] = c; R[:, 1, 2] = -s; R[:, 2, 1] = s; R[:, 2, 2] = c
    elif axis == 'y':
        R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 1, 1] = 1; R[:, 2, 0] = -s; R[:, 2, 2] = c
    else:  # z
        R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1
    return R


# ============================================================================
# MMM Forward Kinematics (vectorized over all T frames)
# ============================================================================

# See previous version for explanation of the joint mapping.
# Mapping: skeleton joint -> (indices in 44-dim angle vec, rotation axes)
_JOINT_DOF = {
    0:  ([3,4,5],    'xyz'),   # BP  pelvis
    1:  ([6,7,8],    'xyz'),   # BT  thorax
    2:  ([0,1,2],    'xyz'),   # BLN lower neck
    3:  ([9,10,11],  'xyz'),   # BUN upper neck
    4:  ([],         ''),      # head tip (leaf)
    5:  ([37,38,39], 'xyz'),   # RS  right shoulder
    6:  ([31,32],    'xz'),    # RE  right elbow
    7:  ([40,41],    'xy'),    # RW  right wrist
    8:  ([21,22,23], 'xyz'),   # LS  left shoulder
    9:  ([15,16],    'xz'),    # LE  left elbow
    10: ([24,25],    'xy'),    # LW  left wrist
    11: ([33,34,35], 'xyz'),   # RH  right hip
    12: ([36],       'x'),     # RK  right knee
    13: ([28,29,30], 'xyz'),   # RA  right ankle
    14: ([42],       'x'),     # RF  right foot
    15: ([43],       'x'),     # RM  right foot rot
    16: ([17,18,19], 'xyz'),   # LH  left hip
    17: ([20],       'x'),     # LK  left knee
    18: ([12,13,14], 'xyz'),   # LA  left ankle
    19: ([26],       'x'),     # LF  left foot
    20: ([27],       'x'),     # LM  left foot rot
}

_PARENTS = [-1, 0, 1, 2, 3, 3, 5, 6, 3, 8, 9, 0, 11, 12, 13, 14, 0, 16, 17, 18, 19]

# Bone directions (Y-up T-pose) matching kit_raw_offsets
_BONE_DIRS = np.array(kit_raw_offsets, dtype=np.float64)

_BONE_FRACS = np.array([
    0.000, 0.100, 0.100, 0.050, 0.080,
    0.105, 0.186, 0.146, 0.105, 0.186, 0.146,
    0.050, 0.245, 0.246, 0.080, 0.040,
    0.050, 0.245, 0.246, 0.080, 0.040,
], dtype=np.float64)

NUM_JOINTS = 21

# Topological order (BFS from root) — just 0..20 since PARENTS[j] < j
_TOPO_ORDER = list(range(NUM_JOINTS))


def _batch_local_rotation(joint_angles, joint_idx):
    """Compute (T, 3, 3) local rotation for a joint across all frames.
    joint_angles: (T, 44).  Fully vectorized."""
    dof_idx, axes = _JOINT_DOF[joint_idx]
    T = len(joint_angles)
    if not dof_idx:
        return np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    R = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    for i, ax in zip(dof_idx, axes):
        Ri = _batch_axis_rotmat(joint_angles[:, i], ax)
        R = np.einsum('tij,tjk->tik', R, Ri)
    return R


def fk_vectorized(root_pos, root_rot_euler, joint_angles, height_m):
    """
    Vectorised FK for all T frames.

    Args:
        root_pos:       (T, 3) root position in mm, MMM Z-up
        root_rot_euler: (T, 3) root Euler angles
        joint_angles:   (T, 44) joint angles in radians
        height_m:       scalar, subject body height

    Returns:
        positions: (T, 21, 3) in metres, Y-up
    """
    T = len(root_pos)
    height_mm = height_m * 1000.0
    bone_offsets = _BONE_DIRS * (_BONE_FRACS * height_mm)[:, None]  # (21, 3)

    # Pre-compute all local rotations for each joint: list of (T, 3, 3)
    local_rots = [_batch_local_rotation(joint_angles, j) for j in range(NUM_JOINTS)]

    # Root world rotation
    R_root = _batch_euler_to_rotmat(root_rot_euler)  # (T, 3, 3)

    positions = np.zeros((T, NUM_JOINTS, 3), dtype=np.float64)
    cum_rot = [None] * NUM_JOINTS  # each (T, 3, 3)

    positions[:, 0] = root_pos
    cum_rot[0] = np.einsum('tij,tjk->tik', R_root, local_rots[0])

    for j in range(1, NUM_JOINTS):
        p = _PARENTS[j]
        # bone offset rotated by parent's cumulative rotation
        offset_world = np.einsum('tij,j->ti', cum_rot[p], bone_offsets[j])
        positions[:, j] = positions[:, p] + offset_world
        cum_rot[j] = np.einsum('tij,tjk->tik', cum_rot[p], local_rots[j])

    # MMM Z-up -> Y-up: x, z, -y
    pos_yup = np.stack([positions[:, :, 0],
                        positions[:, :, 2],
                       -positions[:, :, 1]], axis=-1)
    pos_yup /= 1000.0  # mm -> m
    return pos_yup.astype(np.float64)


# ============================================================================
# MMM XML Parsing
# ============================================================================

def parse_mmm_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    xm = root.findall("Motion")
    if not xm:
        raise RuntimeError(f"No <Motion> in {path}")
    xm = xm[0]

    mpc = xm.find("ModelProcessorConfig")
    height_m = 1.75
    if mpc is not None:
        h = mpc.find("Height")
        if h is not None and h.text:
            height_m = float(h.text.strip())

    xjo = xm.find("JointOrder")
    if xjo is None:
        raise RuntimeError("No <JointOrder>")
    joint_names = [j.get("name") for j in xjo.findall("Joint")]

    xfs = xm.find("MotionFrames")
    if xfs is None:
        raise RuntimeError("No <MotionFrames>")

    rp, rr, ja = [], [], []
    for xf in xfs.findall("MotionFrame"):
        rp.append(_pf(xf.find("RootPosition")))
        rr.append(_pf(xf.find("RootRotation")))
        ja.append(_pf(xf.find("JointPosition")))
    return (joint_names, height_m,
            np.array(rp, np.float64), np.array(rr, np.float64), np.array(ja, np.float64))

def _pf(e):
    return [float(x) for x in e.text.strip().split()] if e is not None and e.text else []


# ============================================================================
# 251-dim feature extraction  (uses existing Skeleton + quaternion utils)
# ============================================================================

# KIT-specific constants (from paramUtil.py)
KIT_L_IDX1, KIT_L_IDX2 = 17, 18
KIT_FID_R, KIT_FID_L = [14, 15], [19, 20]
KIT_FACE_JOINT_IDX = [11, 16, 5, 8]  # r_hip, l_hip, sdr_r, sdr_l
KIT_R_HIP, KIT_L_HIP = 11, 16
KIT_JOINTS_NUM = 21

n_raw_offsets = torch.from_numpy(kit_raw_offsets).float()
kinematic_chain = kit_kinematic_chain


def uniform_skeleton(positions, tgt_offsets):
    """Re-target motion to the reference skeleton. positions: (T, 21, 3) numpy."""
    positions = torch.from_numpy(positions).float()
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(positions[0])

    src_leg_len = (src_offset[KIT_L_IDX1].abs().max() +
                   src_offset[KIT_L_IDX2].abs().max()).item()
    tgt_leg_len = (tgt_offsets[KIT_L_IDX1].abs().max() +
                   tgt_offsets[KIT_L_IDX2].abs().max()).item()

    if src_leg_len < 1e-6:
        scale_rt = 1.0
    else:
        scale_rt = tgt_leg_len / src_leg_len

    tgt_root_pos = positions[:, 0] * scale_rt

    quat_params = src_skel.inverse_kinematics_np(
        positions.numpy(), KIT_FACE_JOINT_IDX)

    src_skel.set_offset(tgt_offsets)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos.numpy())
    return new_joints  # (T, 21, 3) numpy


def process_file(positions, feet_thre):
    """
    Mirrors motGPT/data/humanml/scripts/motion_process.py  process_file()
    but for KIT (21 joints).

    positions: (T, 21, 3) numpy   (already on the reference skeleton)
    Returns:  data (T-1, 251), global_positions, positions, l_velocity
    """
    face_joint_indx = KIT_FACE_JOINT_IDX
    fid_r, fid_l = KIT_FID_R, KIT_FID_L

    # --- Put on Floor ---
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # --- XZ at origin ---
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # --- All initially face Z+ ---
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    global_positions = positions.copy()

    # --- Foot Contacts ---
    def foot_detect(positions, thres):
        velfactor = np.array([thres, thres])
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float64)
        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float64)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    # --- Quaternion IK -> cont6d ---
    r_rot = None

    def get_rifke(positions):
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot_ = quat_params[:, 0].copy()
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot_[1:], velocity)
        r_velocity = qmul_np(r_rot_[1:], qinv_np(r_rot_[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot_

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    # --- Root ---
    root_y = positions[:, 0, 1:2]
    r_velocity_y = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)

    # --- Joint rotation (cont6d, exclude root) ---
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # --- Joint positions (root-relative) ---
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # --- Joint velocity (local frame) ---
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = np.concatenate([root_data, ric_data[:-1], rot_data[:-1],
                           local_vel, feet_l, feet_r], axis=-1)
    return data, global_positions, positions, l_velocity


# ============================================================================
# Annotation handling
# ============================================================================

def load_annotations(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def annotations_to_txt(annotations):
    lines = []
    for ann in annotations:
        ann = ann.strip()
        if not ann: continue
        tokens = " ".join(f"{w}/NOUN" for w in ann.split())
        lines.append(f"{ann}#{tokens}#0.0#0.0")
    return "\n".join(lines)


# ============================================================================
# Splits
# ============================================================================

def create_splits(all_ids, train_r=0.8, val_r=0.1, seed=42):
    rng = np.random.RandomState(seed)
    ids = sorted(all_ids); rng.shuffle(ids)
    n = len(ids); nt = int(n*train_r); nv = int(n*val_r)
    return ids[:nt], ids[nt:nt+nv], ids[nt+nv:]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="KIT-ML -> 251-dim features")
    parser.add_argument('--raw_dir',    type=str, default='/data/jackieye/KIT-ML/raw/')
    parser.add_argument('--output_dir', type=str, default='datasets/kit/')
    parser.add_argument('--min_frames', type=int, default=24)
    parser.add_argument('--ds_num',     type=int, default=8,
                        help='Downsample factor for joint positions (100fps/8=12.5fps, matching MotionGPT3)')
    parser.add_argument('--feet_thre',  type=float, default=0.05)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip motions that already have output .npy files')
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.output_dir
    vec_dir    = os.path.join(out_dir, 'new_joint_vecs')
    joints_dir = os.path.join(out_dir, 'new_joints')
    text_dir   = os.path.join(out_dir, 'texts')
    for d in [vec_dir, joints_dir, text_dir]:
        os.makedirs(d, exist_ok=True)

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith('_mmm.xml'))
    basenames = sorted(set(f.replace('_mmm.xml', '') for f in files))
    print(f"Found {len(basenames)} motions in {raw_dir}")

    # ---- Compute reference skeleton offsets from kit_tgt_skel_id ----
    # We need a reference pose to get target offsets.
    # Use the first successfully parsed motion as reference (or kit_tgt_skel_id if available).
    tgt_skel_path = os.path.join(raw_dir, f"{kit_tgt_skel_id}_mmm.xml")
    if os.path.exists(tgt_skel_path):
        ref_basename = kit_tgt_skel_id
    else:
        ref_basename = basenames[0]
    print(f"Using {ref_basename} as reference skeleton")

    jn, hm, rp, rr, ja = parse_mmm_xml(os.path.join(raw_dir, f"{ref_basename}_mmm.xml"))
    ref_pos = fk_vectorized(rp, rr, ja, hm)       # (T, 21, 3)
    ref_pos_t = torch.from_numpy(ref_pos[0]).float()
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(ref_pos_t)
    print(f"Reference skeleton offsets computed, leg lengths: "
          f"L1={tgt_offsets[KIT_L_IDX1].abs().max():.4f}, "
          f"L2={tgt_offsets[KIT_L_IDX2].abs().max():.4f}")

    all_features = []
    valid_ids: List[str] = []
    ref_jnames = None
    n_skip, n_err = 0, 0

    for basename in tqdm(basenames, desc="Processing KIT-ML"):
        out_vec = os.path.join(vec_dir, f"{basename}.npy")
        out_jnt = os.path.join(joints_dir, f"{basename}.npy")

        if args.skip_existing and os.path.exists(out_vec) and os.path.exists(out_jnt):
            # Still need features for mean/std computation
            try:
                feat = np.load(out_vec)
                if len(feat) >= args.min_frames and not np.isnan(feat).any():
                    all_features.append(feat)
                    valid_ids.append(basename)
            except:
                pass
            continue

        mmm_path = os.path.join(raw_dir, f"{basename}_mmm.xml")
        ann_path = os.path.join(raw_dir, f"{basename}_annotations.json")

        try:
            jnames, height_m, rpos, rrot, jangles = parse_mmm_xml(mmm_path)

            if ref_jnames is None:
                ref_jnames = jnames
            elif jnames != ref_jnames:
                n_skip += 1; continue
            if jangles.shape[1] != 44:
                n_skip += 1; continue

            # FK: angles -> 3D positions  (vectorized)
            pos_3d = fk_vectorized(rpos, rrot, jangles, height_m)  # (T, 21, 3)

            # Re-target to reference skeleton
            pos_3d = uniform_skeleton(pos_3d, tgt_offsets)

            # Downsample joint positions (100fps -> 12.5fps, matching MotionGPT3/HumanML3D KIT pipeline)
            if args.ds_num > 1:
                pos_3d = pos_3d[::args.ds_num]

            # Extract 251-dim features (uses Skeleton IK internally)
            features, gpos, _, l_vel = process_file(pos_3d, args.feet_thre)

            if len(features) < args.min_frames:
                n_skip += 1; continue
            if np.isnan(features).any():
                n_err += 1; continue

            np.save(out_vec, features.astype(np.float32))
            np.save(out_jnt, pos_3d.astype(np.float32))

            annotations = []
            if os.path.exists(ann_path):
                annotations = load_annotations(ann_path)
            txt = annotations_to_txt(annotations) if annotations else ""
            with open(os.path.join(text_dir, f"{basename}.txt"), 'w') as f:
                f.write(txt + "\n")

            all_features.append(features)
            valid_ids.append(basename)

        except Exception as e:
            n_err += 1
            if n_err <= 10:
                tqdm.write(f"  [ERROR] {basename}: {e}")

    print(f"\nProcessed: {len(valid_ids)} | Skipped: {n_skip} | Errors: {n_err}")
    if not valid_ids:
        print("No valid motions. Aborting."); return

    # Mean & Std
    combined = np.concatenate(all_features, axis=0)
    mean = combined.mean(axis=0).astype(np.float32)
    std  = np.maximum(combined.std(axis=0), 1e-6).astype(np.float32)
    np.save(os.path.join(out_dir, 'Mean.npy'), mean)
    np.save(os.path.join(out_dir, 'Std.npy'), std)
    print(f"Feature dimension: {mean.shape[0]}")

    tr, va, te = create_splits(valid_ids, args.train_ratio, args.val_ratio, args.seed)
    for name, ids in [('train.txt', tr), ('val.txt', va), ('test.txt', te), ('all.txt', valid_ids)]:
        with open(os.path.join(out_dir, name), 'w') as f:
            f.write('\n'.join(ids) + '\n')
        print(f"  {name}: {len(ids)}")

    print(f"\nDone -> {out_dir}")
    print(f"Dim: {mean.shape[0]} (expect 251)")
    print(f"Total: {len(valid_ids)} | Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")


if __name__ == '__main__':
    main()
