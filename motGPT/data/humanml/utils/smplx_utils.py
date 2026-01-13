import torch
import os


from smplx import SMPLX
smplx_model_path = '/home/zbf/Desktop/MotionGPT/deps/smpl_models/smplx/SMPLX_NEUTRAL.npz'
def process_pose(pose):
    device = pose.device
    smplx_model = SMPLX(smplx_model_path, num_betas=10, use_pca=False, use_face_contour=True, batch_size=1).to(device)
    # pose = torch.from_numpy(pose).float().cuda()
    # print(pose.shape)
    # exit()
    
    bs, num_frames = pose.shape[:2]
    pose = pose.reshape(bs*num_frames, 322)

    param = {
        'root_orient': pose[..., :3],  # controls the global root orientation
        'pose_body': pose[..., 3:3+63],  # controls the body
        'pose_hand': pose[..., 66:66+90],  # controls the finger articulation
        'pose_jaw': pose[..., 66+90:66+93],  # controls the yaw pose
        'face_expr': pose[..., 159:159+50],  # controls the face expression
        'face_shape': pose[..., 209:209+100],  # controls the face shape
        'trans': pose[..., 309:309+3],  # controls the global body position
        'betas': pose[..., 312:],  # controls the body shape. Body shape is static
    }

    batch_size = param['face_expr'].shape[0]
    zero_pose = torch.zeros((batch_size, 3)).float().cuda()

    smplx_output = smplx_model(betas=param['betas'], body_pose=param['pose_body'],
                                global_orient=param['root_orient'], pose2rot=True, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                                left_hand_pose=param['pose_hand'][..., :45], right_hand_pose=param['pose_hand'][..., 45:],
                                expression=param['face_expr'][..., :10], transl=param['trans'])
                        
    # vertices = smplx_output.vertices.reshape(bs, num_frames, 10475, 3)
    joints = smplx_output.joints.reshape(bs, num_frames, -1, 3)
    # joints = joints[:, joint_idx, :]
    # print(joints.shape)  # 32, 136, 144, 3
    # exit()
    return joints.to(device)