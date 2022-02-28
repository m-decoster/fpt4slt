import numpy as np
import torch
from scipy.spatial.transform import Rotation


def remove_keypoints_and_normalize(keypoint_tokens):
    """This pre-processing function removes certain keypoints and normalizes the poses
    so that the shoulder length is 1, and so that the upper bodies are oriented towards the camera.

    :param keypoint_tokens: The tokenized keypoint features.
    """
    # First remove unwanted keypoints.
    keypoint_tokens = _remove_unwanted_keypoints(keypoint_tokens)

    # Now normalize.
    keypoint_tokens = _normalize_keypoints(keypoint_tokens)

    return torch.tensor(keypoint_tokens, dtype=torch.float)


def _normalize_keypoints(keypoint_tokens):
    """Normalizes the keypoints.
    First, we compute the distance between the shoulders in the first frame and divide all keypoints in all
    poses in the sequence by this distance.
    Secondly, compute the angle needed to rotate the pose such that the upper body
    directly faces the camera (in the first frame). Then, we rotate all poses in the sequence by this angle.
    Finally, we translate all keypoints in all frames based on the offset from the origin in the first frame.
    This way, we obtain a normalized starting point but allow for
    temporal movement in 3D space to affect the features.

    The origin is chosen here as the center between the two shoulders (MediaPipe Pose defines it as the center between
    the hips but we don't have the hips in our sign language data).

    :param keypoint_tokens: The tokenized keypoint features after removal.
    :returns: The normalized keypoint tokens."""
    first_frame = keypoint_tokens[0]

    # Pose indices 11 and 12: see https://google.github.io/mediapipe/solutions/pose.html, more specifically:
    # https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png
    left_shoulder = first_frame[11 * 3:12 * 3]
    right_shoulder = first_frame[12 * 3:13 * 3]
    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
    origin = 0.5 * (right_shoulder + left_shoulder)

    # Scaling.
    for frame_index in range(len(keypoint_tokens)):
        keypoint_tokens[frame_index] /= shoulder_distance
    origin /= shoulder_distance

    # Rotation.
    left_shoulder = keypoint_tokens[0][11 * 3:12 * 3]
    right_shoulder = keypoint_tokens[0][12 * 3:13 * 3]
    yaw = np.dot([0.0, 1.0, 0.0], left_shoulder - right_shoulder)
    # TODO: what about the roll?
    rotation_matrix = Rotation.from_euler('xyz', [0.0, yaw, 0.0], degrees=False).as_matrix()
    for frame_index in range(len(keypoint_tokens)):
        pose = keypoint_tokens[frame_index]
        x = pose[0::3]
        y = pose[1::3]
        z = pose[2::3]
        xyz_pose = np.stack((x, y, z), axis=1)
        xyz_pose_rotated = xyz_pose @ rotation_matrix
        x = xyz_pose_rotated[:, 0]
        y = xyz_pose_rotated[:, 1]
        z = xyz_pose_rotated[:, 2]
        keypoint_tokens[frame_index][0::3] = x
        keypoint_tokens[frame_index][1::3] = y
        keypoint_tokens[frame_index][2::3] = z

    # Translation.
    left_shoulder = keypoint_tokens[0][11 * 3:12 * 3]
    right_shoulder = keypoint_tokens[0][12 * 3:13 * 3]
    origin = 0.5 * (right_shoulder + left_shoulder)
    for frame_index in range(len(keypoint_tokens)):
        keypoint_tokens[frame_index][0::3] -= origin[0]
        keypoint_tokens[frame_index][1::3] -= origin[1]
        keypoint_tokens[frame_index][2::3] -= origin[2]

    return keypoint_tokens


def _remove_unwanted_keypoints(keypoint_tokens):
    """Removes unwanted keypoints (defined inside this function).

    :param keypoint_tokens: A list of keypoint poses.
    :returns: A new list of keypoint poses with fewer keypoints."""
    # These are the keypoints we want to keep.
    # Pose: upper body only.
    POSE_INDICES_TO_KEEP = list(range(23))
    # Face: eyebrows and mouth.
    FACE_INDICES_TO_KEEP = [0, 13, 14, 17, 37, 39, 40, 46, 52, 53, 55, 61, 63, 65, 66, 70, 78, 80, 81, 82, 84, 87, 88,
                            91, 95, 105, 107, 146, 178, 181, 185, 191, 267, 269, 270, 276, 282, 283, 285, 291, 293, 295,
                            296, 300, 308, 310, 311, 312, 314, 317, 318, 321, 324, 334, 336, 375, 402, 405, 409, 415]
    # Hands: all keypoints.
    HAND_INDICES_TO_KEEP = list(range(21))
    # These are the keypoint indices in the entire array.
    POSE_INDICES = np.arange(0, 33)
    FACE_INDICES = np.arange(33, 33 + 468)
    LEFT_HAND_INDICES = np.arange(FACE_INDICES[-1] + 1, FACE_INDICES[-1] + 1 + 21)
    RIGHT_HAND_INDICES = np.arange(LEFT_HAND_INDICES[-1] + 1, LEFT_HAND_INDICES[-1] + 1 + 21)
    # Now, for every frame, first remove the keypoints.
    new_keypoint_tokens = []
    for pose in keypoint_tokens:
        x = pose[0::3]
        y = pose[1::3]
        z = pose[2::3]
        body_x, body_y, body_z = _keep_only_wanted_keypoints(POSE_INDICES, POSE_INDICES_TO_KEEP, x, y, z)
        face_x, face_y, face_z = _keep_only_wanted_keypoints(FACE_INDICES, FACE_INDICES_TO_KEEP, x, y, z)
        left_hand_x, left_hand_y, left_hand_z = _keep_only_wanted_keypoints(LEFT_HAND_INDICES,
                                                                            HAND_INDICES_TO_KEEP, x, y, z)
        right_hand_x, right_hand_y, right_hand_z = _keep_only_wanted_keypoints(RIGHT_HAND_INDICES,
                                                                               HAND_INDICES_TO_KEEP, x, y, z)
        # Interleave the body part keypoints.
        body = np.empty((body_x.shape[0] * 3,))
        body[0::3], body[1::3], body[2::3] = body_x, body_y, body_z
        face = np.empty((face_x.shape[0] * 3), )
        face[0::3], face[1::3], face[2::3] = face_x, face_y, face_z
        left_hand = np.empty((left_hand_x.shape[0] * 3), )
        left_hand[0::3], left_hand[1::3], left_hand[2::3] = left_hand_x, left_hand_y, left_hand_z
        right_hand = np.empty((right_hand_x.shape[0] * 3), )
        right_hand[0::3], right_hand[1::3], right_hand[2::3] = right_hand_x, right_hand_y, right_hand_z
        # Concatenate
        new_pose = np.concatenate([body, face, left_hand, right_hand])
        new_keypoint_tokens.append(new_pose)
    return new_keypoint_tokens


def _keep_only_wanted_keypoints(body_part_indices, indices_to_keep, x, y, z):
    """Creates new keypoint arrays keeping only the wanted indices.

    :param body_part_indices: The indices that are used to select only the body part from the keypoint arrays.
    :param indices_to_keep: The  indices (in the body part arrays) that we wish to keep.
    :param x: The x coordinates of the entire keypoint array.
    :param y: The y coordinates of the entire keypoint array.
    :param z: The z coordinates of the entire keypoint array.
    :returns: The new x, y and z coordinates (as a tuple)."""
    # The *_TO_KEEP indices are 0-based in the body part arrays, so we first need to select those.
    part_x, part_y, part_z = x[body_part_indices], y[body_part_indices], z[body_part_indices]
    # Now we can remove the unwanted indices by keeping the ones we do want.
    new_x, new_y, new_z = part_x[indices_to_keep], part_y[indices_to_keep], part_z[indices_to_keep]
    return new_x, new_y, new_z
