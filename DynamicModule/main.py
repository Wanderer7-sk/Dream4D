import argparse
import json
from PIL import Image
import numpy as np
import torch

from demo.dynamic import DynamicModule

def load_camera_pose_type(camera_pose_meta_path: str):
    with open(camera_pose_meta_path, "r") as f:
        data = json.load(f)

    return list(data.keys())

def main():
    parser = argparse.ArgumentParser(description='Command Line Interface')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--model_name', type=str, required=True, help='Model name to use')
    parser.add_argument('--camera_pose_type', type=str, default='stationary', 
                       choices=load_camera_pose_type('./demo/camera_poses.json'),
                       help='Camera pose type')
    parser.add_argument('--negative_prompt', type=str, 
                       default="Fast movement, jittery motion, abrupt transitions, distorted body, missing limbs, unnatural posture, blurry, cropped, extra limbs, bad anatomy, deformed, glitchy motion, artifacts.",
                       help='Negative prompt')
    parser.add_argument('--trace_extract_ratio', type=float, default=0.1, help='Trace extract ratio')
    parser.add_argument('--trace_scale_factor', type=float, default=1.0, help='Camera trace scale factor')
    parser.add_argument('--frame_stride', type=int, default=2, help='Frame stride')
    parser.add_argument('--steps', type=int, default=25, help='Sampling steps')
    parser.add_argument('--camera_cfg', type=float, default=1.0, help='Camera CFG')
    parser.add_argument('--cfg_scale', type=float, default=5.5, help='CFG Scale')
    parser.add_argument('--seed', type=int, default=12333, help='Random seed')
    parser.add_argument('--enable_camera_condition', action='store_true', default=True, 
                       help='Enable camera condition')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--result_dir', type=str, default='./demo/results', 
                       help='Directory to save results')
    parser.add_argument('--model_meta_path', type=str, default='./demo/models.json', 
                       help='Path to model metadata')
    parser.add_argument('--camera_pose_meta_path', type=str, default='./demo/camera_poses.json', 
                       help='Path to camera pose metadata')

    args = parser.parse_args()

    # Initialize DynamicModule
    dynamic_module = DynamicModule(
        result_dir=args.result_dir,
        model_meta_path=args.model_meta_path,
        camera_pose_meta_path=args.camera_pose_meta_path,
        device=args.device
    )

    # Load input image
    input_image = np.array(Image.open(args.input_image).convert('RGB'))

    # Generate video
    print("Generating video...")
    result = dynamic_module.get_image(
        model_name=args.model_name,
        ref_img=input_image,
        caption=args.prompt,
        negative_prompt=args.negative_prompt,
        camera_pose_type=args.camera_pose_type,
        input_image_path=args.input_image,
        trace_extract_ratio=args.trace_extract_ratio,
        frame_stride=args.frame_stride,
        steps=args.steps,
        trace_scale_factor=args.trace_scale_factor,
        camera_cfg=args.camera_cfg,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        enable_camera_condition=args.enable_camera_condition
    )

    print(f"Video generated and saved to: {result[0]}")
    if len(result) > 1:
        print(f"Camera trajectory saved to: {result[1]}")

if __name__ == "__main__":
    main() 