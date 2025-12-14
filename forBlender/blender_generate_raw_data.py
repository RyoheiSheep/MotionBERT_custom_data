import bpy
import bpy_extras
import os
import numpy as np
import pickle
from math import pi, sin, cos, tan
from pathlib import Path
from datetime import datetime
import platform

# ================= Config =================
class Config:
    # Root
    DATA_ROOT = Path("D:/MotionBERT_Data/motion3d/raw_blender")

    # Motion
    MOTION_NAME = "Walking"
    ARMATURE_NAME = "Armature"

    # Camera
    NUM_CAMS_PER_CIRCLE = 8
    SCREEN_RATIOS = [0.6, 0.8]
    CAMERA_HEIGHT = 1.2
    RESOLUTION = (1000, 1000)

    # Auto Run ID
    RUN_ID = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")

    @classmethod
    def output_dir(cls):
        path = cls.DATA_ROOT / cls.MOTION_NAME / cls.RUN_ID
        path.mkdir(parents=True, exist_ok=True)
        return path

# ================= Skeleton =================
BONE_MAP = {
    "Pelvis": "Hips", "R_Hip": "RightUpLeg", "R_Knee": "RightLeg", "R_Ankle": "RightFoot",
    "L_Hip": "LeftUpLeg", "L_Knee": "LeftLeg", "L_Ankle": "LeftFoot",
    "Torso": "Spine", "Neck": "Neck", "Nose": "Head", "Head": "Head_End",
    "L_Shoulder": "LeftShoulder", "L_Elbow": "LeftArm", "L_Wrist": "LeftForeArm",
    "R_Shoulder": "RightShoulder", "R_Elbow": "RightArm", "R_Wrist": "RightForeArm"
}

H36M_ORDER = [
    "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle",
    "Torso", "Neck", "Nose", "Head",
    "L_Shoulder", "L_Elbow", "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist"
]

# ================= Core Logic =================

def get_bone_positions(armature, frame_start, frame_end):
    scene = bpy.context.scene
    bone_data = []

    target_bones = []
    for name in H36M_ORDER:
        bname = BONE_MAP.get(name)
        target_bones.append(armature.pose.bones.get(bname, armature.pose.bones[0]))

    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        frame_pos = []
        for bone in target_bones:
            pos = armature.matrix_world @ bone.head
            frame_pos.append([pos.x * 1000, pos.y * 1000, pos.z * 1000])
        bone_data.append(frame_pos)

    return np.array(bone_data, dtype=np.float32)


def project_to_pixel(scene, camera, positions_3d_mm):
    T, J, _ = positions_3d_mm.shape
    out = np.zeros((T, J, 3), dtype=np.float32)
    rx, ry = scene.render.resolution_x, scene.render.resolution_y

    for t in range(T):
        for j in range(J):
            vec = bpy.mathutils.Vector(positions_3d_mm[t, j] / 1000)
            ndc = bpy_extras.object_utils.world_to_camera_view(scene, camera, vec)
            px = ndc.x * rx
            py = (1 - ndc.y) * ry
            conf = 1.0 if 0 <= ndc.x <= 1 and 0 <= ndc.y <= 1 else 0.0
            out[t, j] = [px, py, conf]

    return out


def save_meta(scene, output_dir):
    meta = {
        "motion": Config.MOTION_NAME,
        "armature": Config.ARMATURE_NAME,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "resolution": Config.RESOLUTION,
        "num_cams": Config.NUM_CAMS_PER_CIRCLE * len(Config.SCREEN_RATIOS),
        "blender_file": bpy.data.filepath,
        "blender_version": bpy.app.version_string,
        "platform": platform.platform(),
        "created_at": datetime.now().isoformat()
    }

    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)


def main():
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = Config.RESOLUTION

    armature = bpy.data.objects[Config.ARMATURE_NAME]
    output_dir = Config.output_dir()

    save_meta(scene, output_dir)

    gt_3d = get_bone_positions(armature, scene.frame_start, scene.frame_end)

    min_z, max_z = np.min(gt_3d[:, :, 2]), np.max(gt_3d[:, :, 2])
    center = armature.location
    fov = 50 / 180 * pi
    base_dist = ((max_z - min_z) / 1000 / 2) / tan(fov / 2)

    cam_idx = 0
    for ratio in Config.SCREEN_RATIOS:
        dist = base_dist / ratio
        for i in range(Config.NUM_CAMS_PER_CIRCLE):
            angle = 2 * pi * i / Config.NUM_CAMS_PER_CIRCLE
            cam = bpy.data.cameras.new("TempCam")
            cam.angle = fov
            cam_obj = bpy.data.objects.new("TempCam", cam)
            bpy.context.collection.objects.link(cam_obj)

            cam_obj.location = (
                center.x + dist * cos(angle),
                center.y + dist * sin(angle),
                Config.CAMERA_HEIGHT
            )

            cam_obj.rotation_euler = (center - cam_obj.location).to_track_quat('-Z', 'Y').to_euler()
            bpy.context.view_layer.update()

            input_2d = project_to_pixel(scene, cam_obj, gt_3d)

            data = {
                "data_label": gt_3d,
                "data_input": input_2d,
                "res_w": Config.RESOLUTION[0],
                "res_h": Config.RESOLUTION[1]
            }

            with open(output_dir / f"raw_cam{cam_idx:02d}.pkl", "wb") as f:
                pickle.dump(data, f)

            bpy.data.objects.remove(cam_obj)
            cam_idx += 1

    print("Raw generation complete.")


if __name__ == "__main__":
    main()