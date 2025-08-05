import copy
import json
import os
import os.path as osp
import queue
import secrets
import threading
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Literal

import gradio as gr
import httpx
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import viser
import viser.transforms as vt
from einops import rearrange
from gradio import networking
from gradio.context import LocalContext
from gradio.tunneling import CERTIFICATE_PATH, Tunnel

from static.eval import (
    IS_TORCH_NIGHTLY,
    chunk_input_and_test,
    create_transforms_simple,
    infer_prior_stats,
    run_one_scene,
    transform_img_and_K,
)
from static.geometry import (
    DEFAULT_FOV_RAD,
    get_default_intrinsics,
    get_preset_pose_fov,
    normalize_scene,
)
from static.gui import define_gui
from static.model import SGMWrapper
from static.modules.autoencoder import AutoEncoder
from static.modules.conditioner import CLIPConditioner
from static.modules.preprocessor import Dust3rPipeline
from static.sampling import DiscreteDenoiser
from utils import load_model

device = "cuda:0"

WORK_DIR = "work_dirs/demo_gr"
MAX_SESSIONS = 1
ADVANCE_EXAMPLE_MAP = [
    ("assets/advance/blue-car.jpg", ["assets/advance/blue-car.jpg"]),
    ("assets/advance/garden-4_0.jpg", [
        "assets/advance/garden-4_0.jpg",
        "assets/advance/garden-4_1.jpg",
        "assets/advance/garden-4_2.jpg",
        "assets/advance/garden-4_3.jpg",
    ]),
    ("assets/advance/vgg-lab-4_0.png", [
        "assets/advance/vgg-lab-4_0.png",
        "assets/advance/vgg-lab-4_1.png",
        "assets/advance/vgg-lab-4_2.png",
        "assets/advance/vgg-lab-4_3.png",
    ]),
    ("assets/advance/telebooth-2_0.jpg", [
        "assets/advance/telebooth-2_0.jpg",
        "assets/advance/telebooth-2_1.jpg",
    ]),
    ("assets/advance/backyard-7_0.jpg", [
        "assets/advance/backyard-7_0.jpg",
        "assets/advance/backyard-7_1.jpg",
        "assets/advance/backyard-7_2.jpg",
        "assets/advance/backyard-7_3.jpg",
        "assets/advance/backyard-7_4.jpg",
        "assets/advance/backyard-7_5.jpg",
        "assets/advance/backyard-7_6.jpg",
    ]),
]

if IS_TORCH_NIGHTLY:
    COMPILE = True
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
else:
    COMPILE = False

# Shared global objects
DUST3R = Dust3rPipeline(device=device)
MODEL = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
AE = AutoEncoder(chunk_size=1).to(device)
CONDITIONER = CLIPConditioner().to(device)
DENOISER = DiscreteDenoiser(num_idx=1000, device=device)
VERSION_DICT = {"H": 576, "W": 576, "T": 21, "C": 4, "f": 8, "options": {}}
SERVERS = {}
ABORT_EVENTS = {}

if COMPILE:
    MODEL = torch.compile(MODEL)
    CONDITIONER = torch.compile(CONDITIONER)
    AE = torch.compile(AE)

class StaticRenderer:
    """Renderer for static virtual camera demo."""

    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.gui_state = None

    def preprocess(self, img_input):
        """Preprocess input images and extract camera parameters."""
        shorter = 576
        shorter = round(shorter / 64) * 64
        if isinstance(img_input, str):
            # Single image mode
            img = torch.as_tensor(iio.imread(img_input) / 255.0, dtype=torch.float32)[None, ..., :3]
            img = transform_img_and_K(img.permute(0, 3, 1, 2), shorter, K=None, size_stride=64)[0].permute(0, 2, 3, 1)
            K = get_default_intrinsics(aspect_ratio=img.shape[2] / img.shape[1])
            c2ws = torch.eye(4)[None]
            time.sleep(0.1)
            return (
                {"input_imgs": img, "input_Ks": K, "input_c2ws": c2ws, "input_wh": (img.shape[2], img.shape[1]),
                 "points": [np.zeros((0, 3))], "point_colors": [np.zeros((0, 3))], "scene_scale": 1.0},
                gr.update(visible=False),
                gr.update(),
            )
        else:
            # Multi-image mode
            img_paths = [p for (p, _) in img_input]
            imgs, Ks, c2ws, points, colors = DUST3R.infer_cameras_and_points(img_paths)
            if len(img_paths) == 1:
                imgs, Ks, c2ws, points, colors = imgs[:1], Ks[:1], c2ws[:1], points[:1], colors[:1]
            imgs = [img[..., :3] for img in imgs]
            point_chunks = [p.shape[0] for p in points]
            point_indices = np.cumsum(point_chunks)[:-1]
            c2ws, points, _ = normalize_scene(c2ws, np.concatenate(points, 0), camera_center_method="poses")
            points = np.split(points, point_indices, 0)
            scene_scale = np.median(np.ptp(np.concatenate([c2ws[:, :3, 3], *points], 0), -1))
            c2ws[:, :3, 3] /= scene_scale
            points = [point / scene_scale for point in points]
            imgs = [torch.as_tensor(img / 255.0, dtype=torch.float32) for img in imgs]
            Ks = torch.as_tensor(Ks)
            c2ws = torch.as_tensor(c2ws)
            new_imgs, new_Ks = [], []
            for img, K in zip(imgs, Ks):
                img = rearrange(img, "h w c -> 1 c h w")
                img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
                K = K / K.new_tensor([img.shape[-1], img.shape[-2], 1])[:, None]
                new_imgs.append(img)
                new_Ks.append(K)
            imgs = torch.cat(new_imgs, 0)
            imgs = rearrange(imgs, "b c h w -> b h w c")[..., :3]
            Ks = torch.cat(new_Ks, 0)
            return (
                {"input_imgs": imgs, "input_Ks": Ks, "input_c2ws": c2ws, "input_wh": (imgs.shape[2], imgs.shape[1]),
                 "points": points, "point_colors": colors, "scene_scale": scene_scale},
                gr.update(visible=False),
                gr.update(),
            )

    def visualize_scene(self, preprocessed):
        """Visualize input cameras and points in the 3D scene."""
        server = self.server
        server.scene.reset()
        server.gui.reset()
        set_bkgd_color(server)
        imgs, Ks, c2ws, wh, points, colors, scale = (
            preprocessed["input_imgs"], preprocessed["input_Ks"], preprocessed["input_c2ws"],
            preprocessed["input_wh"], preprocessed["points"], preprocessed["point_colors"], preprocessed["scene_scale"]
        )
        W, H = wh
        server.scene.set_up_direction(-c2ws[..., :3, 1].mean(0).numpy())
        init_fov = 2 * np.arctan(1 / (2 * Ks[0, 1, 1].item())) if H > W else 2 * np.arctan(1 / (2 * Ks[0, 0, 0].item()))
        init_fov_deg = float(init_fov / np.pi * 180.0)
        frustum_nodes, pcd_nodes = [], []
        for i in range(len(imgs)):
            K = Ks[i]
            frustum = server.scene.add_camera_frustum(
                f"/scene_assets/cameras/{i}",
                fov=2 * np.arctan(1 / (2 * K[1, 1].item())),
                aspect=W / H,
                scale=0.1 * scale,
                image=(imgs[i].numpy() * 255.0).astype(np.uint8),
                wxyz=vt.SO3.from_matrix(c2ws[i, :3, :3].numpy()).wxyz,
                position=c2ws[i, :3, 3].numpy(),
            )
            def get_handler(frustum):
                def handler(event: viser.GuiEvent) -> None:
                    assert event.client_id is not None
                    client = server.get_clients()[event.client_id]
                    with client.atomic():
                        client.camera.position = frustum.position
                        client.camera.wxyz = frustum.wxyz
                        look_direction = vt.SO3(frustum.wxyz).as_matrix()[:, 2]
                        position_origin = -frustum.position
                        client.camera.look_at = (
                            frustum.position
                            + np.dot(look_direction, position_origin)
                            / np.linalg.norm(position_origin)
                            * look_direction
                        )
                return handler
            frustum.on_click(get_handler(frustum))
            frustum_nodes.append(frustum)
            pcd = server.scene.add_point_cloud(
                f"/scene_assets/points/{i}",
                points[i],
                colors[i],
                point_size=0.01 * scale,
                point_shape="circle",
            )
            pcd_nodes.append(pcd)
        with server.gui.add_folder("Scene scale", expand_by_default=False, order=200):
            camera_scale_slider = server.gui.add_slider("Log camera scale", initial_value=0.0, min=-2.0, max=2.0, step=0.1)
            @camera_scale_slider.on_update
            def _(_): [setattr(frustum_nodes[i], 'scale', 0.1 * scale * 10**camera_scale_slider.value) for i in range(len(frustum_nodes))]
            point_scale_slider = server.gui.add_slider("Log point scale", initial_value=0.0, min=-2.0, max=2.0, step=0.1)
            @point_scale_slider.on_update
            def _(_): [setattr(pcd_nodes[i], 'point_size', 0.01 * scale * 10**point_scale_slider.value) for i in range(len(pcd_nodes))]
        self.gui_state = define_gui(server, init_fov=init_fov_deg, img_wh=wh, scene_scale=scale)

    def get_target_c2ws_and_Ks_from_gui(self, preprocessed):
        """Get target camera poses and intrinsics from GUI."""
        W, H = preprocessed["input_wh"]
        gui_state = self.gui_state
        assert gui_state and gui_state.camera_traj_list
        target_c2ws, target_Ks = [], []
        for item in gui_state.camera_traj_list:
            target_c2ws.append(item["w2c"])
            K = np.array(item["K"]).reshape(3, 3) / np.array([W, H, 1])[:, None]
            target_Ks.append(K)
        target_c2ws = torch.as_tensor(np.linalg.inv(np.array(target_c2ws).reshape(-1, 4, 4)))
        target_Ks = torch.as_tensor(np.array(target_Ks).reshape(-1, 3, 3))
        return target_c2ws, target_Ks

    def get_target_c2ws_and_Ks_from_preset(self, preprocessed, preset_traj, num_frames, zoom_factor):
        """Get target camera poses and intrinsics from preset trajectory."""
        img_wh = preprocessed["input_wh"]
        start_c2w = preprocessed["input_c2ws"][0]
        start_w2c = torch.linalg.inv(start_c2w)
        look_at = torch.tensor([0, 0, 10])
        start_fov = DEFAULT_FOV_RAD
        target_c2ws, target_fovs = get_preset_pose_fov(
            preset_traj, num_frames, start_w2c, look_at, -start_c2w[:3, 1], start_fov,
            spiral_radii=[1.0, 1.0, 0.5], zoom_factor=zoom_factor,
        )
        target_c2ws = torch.as_tensor(target_c2ws)
        target_Ks = get_default_intrinsics(target_fovs, aspect_ratio=img_wh[0] / img_wh[1])
        return target_c2ws, target_Ks

    def export_output_data(self, preprocessed, output_dir):
        """Export processed images and camera data for external use."""
        imgs, Ks, c2ws, wh = preprocessed["input_imgs"], preprocessed["input_Ks"], preprocessed["input_c2ws"], preprocessed["input_wh"]
        target_c2ws, target_Ks = self.get_target_c2ws_and_Ks_from_gui(preprocessed)
        num_inputs, num_targets = len(imgs), len(target_c2ws)
        imgs = (imgs.cpu().numpy() * 255.0).astype(np.uint8)
        c2ws, Ks, target_c2ws, target_Ks = map(lambda x: x.cpu().numpy(), [c2ws, Ks, target_c2ws, target_Ks])
        img_whs = np.array(wh)[None].repeat(num_inputs + len(target_Ks), 0)
        os.makedirs(output_dir, exist_ok=True)
        img_paths = []
        for i, img in enumerate(imgs):
            iio.imwrite(img_path := osp.join(output_dir, f"{i:03d}.png"), img)
            img_paths.append(img_path)
        for i in range(num_targets):
            iio.imwrite(img_path := osp.join(output_dir, f"{i + num_inputs:03d}.png"),
                        np.zeros((wh[1], wh[0], 3), dtype=np.uint8))
            img_paths.append(img_path)
        all_c2ws = np.concatenate([c2ws, target_c2ws]) @ np.diag([1, -1, -1, 1])
        all_Ks = np.concatenate([Ks, target_Ks])
        create_transforms_simple(output_dir, img_paths, img_whs, all_c2ws, all_Ks)
        split_dict = {"train_ids": list(range(num_inputs)), "test_ids": list(range(num_inputs, num_inputs + num_targets))}
        with open(osp.join(output_dir, f"train_test_split_{num_inputs}.json"), "w") as f:
            json.dump(split_dict, f, indent=4)
        gr.Info(f"Output data saved to {output_dir}", duration=1)

    def render(self, preprocessed, session_hash, seed, chunk_strategy, cfg, preset_traj, num_frames, zoom_factor, camera_scale):
        """Main rendering pipeline, yields video output and progress."""
        render_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        render_dir = osp.join(WORK_DIR, render_name)
        imgs, Ks, c2ws, (W, H) = preprocessed["input_imgs"], preprocessed["input_Ks"], preprocessed["input_c2ws"], preprocessed["input_wh"]
        num_inputs = len(imgs)
        if preset_traj is None:
            target_c2ws, target_Ks = self.get_target_c2ws_and_Ks_from_gui(preprocessed)
        else:
            assert num_frames is not None and num_inputs == 1
            c2ws = torch.eye(4)[None].to(dtype=c2ws.dtype)
            target_c2ws, target_Ks = self.get_target_c2ws_and_Ks_from_preset(preprocessed, preset_traj, num_frames, zoom_factor)
        all_c2ws = torch.cat([c2ws, target_c2ws], 0)
        all_Ks = torch.cat([Ks, target_Ks], 0) * Ks.new_tensor([W, H, 1])[:, None]
        num_targets = len(target_c2ws)
        input_indices = list(range(num_inputs))
        target_indices = np.arange(num_inputs, num_inputs + num_targets).tolist()
        T = VERSION_DICT["T"]
        version_dict = copy.deepcopy(VERSION_DICT)
        num_anchors = infer_prior_stats(T, num_inputs, num_total_frames=num_targets, version_dict=version_dict)
        T = version_dict["T"]
        anchor_indices = np.linspace(num_inputs, num_inputs + num_targets - 1, num_anchors).tolist()
        anchor_c2ws = all_c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks = all_Ks[[round(ind) for ind in anchor_indices]]
        all_imgs_np = (F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).numpy() * 255.0).astype(np.uint8)
        image_cond = {"img": all_imgs_np, "input_indices": input_indices, "prior_indices": anchor_indices}
        camera_cond = {"c2w": all_c2ws, "K": all_Ks, "input_indices": list(range(num_inputs + num_targets))}
        num_steps = 50
        options = copy.deepcopy(VERSION_DICT["options"])
        options.update({
            "chunk_strategy": chunk_strategy, "video_save_fps": 30.0, "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4, "guider_types": [1, 2], "cfg": [float(cfg), 3.0 if num_inputs >= 9 else 2.0],
            "camera_scale": camera_scale, "num_steps": num_steps, "cfg_min": 1.2, "encoding_t": 1, "decoding_t": 1
        })
        assert session_hash in ABORT_EVENTS
        abort_event = ABORT_EVENTS[session_hash]
        abort_event.clear()
        options["abort_event"] = abort_event
        task = "img2trajvid"
        T_first_pass = T[0] if isinstance(T, (list, tuple)) else T
        chunk_strategy_first_pass = options.get("chunk_strategy_first_pass", "gt-nearest")
        num_chunks_0 = len(chunk_input_and_test(
            T_first_pass, c2ws, anchor_c2ws, input_indices, image_cond["prior_indices"],
            options={**options, "sampler_verbose": False}, task=task, chunk_strategy=chunk_strategy_first_pass,
            gt_input_inds=list(range(c2ws.shape[0])),
        )[1])
        anchor_argsort = np.argsort(input_indices + anchor_indices).tolist()
        anchor_indices = np.array(input_indices + anchor_indices)[anchor_argsort].tolist()
        gt_input_inds = [anchor_argsort.index(i) for i in range(c2ws.shape[0])]
        anchor_c2ws_second_pass = torch.cat([c2ws, anchor_c2ws], dim=0)[anchor_argsort]
        T_second_pass = T[1] if isinstance(T, (list, tuple)) else T
        chunk_strategy = options.get("chunk_strategy", "nearest")
        num_chunks_1 = len(chunk_input_and_test(
            T_second_pass, anchor_c2ws_second_pass, target_c2ws, anchor_indices, target_indices,
            options={**options, "sampler_verbose": False}, task=task, chunk_strategy=chunk_strategy, gt_input_inds=gt_input_inds,
        )[1])
        second_pass_pbar = gr.Progress().tqdm(iterable=None, desc="Second pass sampling", total=num_chunks_1 * num_steps)
        first_pass_pbar = gr.Progress().tqdm(iterable=None, desc="First pass sampling", total=num_chunks_0 * num_steps)
        video_path_generator = run_one_scene(
            task=task,
            version_dict={"H": H, "W": W, "T": T, "C": VERSION_DICT["C"], "f": VERSION_DICT["f"], "options": options},
            model=MODEL, ae=AE, conditioner=CONDITIONER, denoiser=DENOISER,
            image_cond=image_cond, camera_cond=camera_cond, save_path=render_dir,
            use_traj_prior=True, traj_prior_c2ws=anchor_c2ws, traj_prior_Ks=anchor_Ks,
            seed=seed, gradio=True, first_pass_pbar=first_pass_pbar, second_pass_pbar=second_pass_pbar,
            abort_event=abort_event,
        )
        output_queue = queue.Queue()
        blocks = LocalContext.blocks.get()
        event_id = LocalContext.event_id.get()
        def worker():
            LocalContext.blocks.set(blocks)
            LocalContext.event_id.set(event_id)
            for i, video_path in enumerate(video_path_generator):
                if i == 0:
                    output_queue.put((video_path, gr.update(), gr.update(), gr.update()))
                elif i == 1:
                    output_queue.put((video_path, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)))
                else:
                    gr.Error("More than two passes during rendering.")
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while thread.is_alive() or not output_queue.empty():
            if abort_event.is_set():
                thread.join()
                abort_event.clear()
                yield (gr.update(), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
            time.sleep(0.1)
            while not output_queue.empty():
                yield output_queue.get()

def setup_tunnel(local_host, local_port, share_token, share_server_address):
    """Setup a tunnel for remote access."""
    share_server_address = networking.GRADIO_SHARE_SERVER_ADDRESS if share_server_address is None else share_server_address
    if share_server_address is None:
        try:
            response = httpx.get(networking.GRADIO_API_SERVER, timeout=30)
            payload = response.json()[0]
            remote_host, remote_port = payload["host"], int(payload["port"])
            certificate = payload["root_ca"]
            Path(CERTIFICATE_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(CERTIFICATE_PATH, "w") as f:
                f.write(certificate)
        except Exception as e:
            raise RuntimeError("Could not get share link from Gradio API Server.") from e
    else:
        remote_host, remote_port = share_server_address.split(":")
        remote_port = int(remote_port)
    tunnel = Tunnel(remote_host, remote_port, local_host, local_port, share_token)
    address = tunnel.start_tunnel()
    return address, tunnel

def set_bkgd_color(server):
    """Set background color for the 3D scene."""
    server.scene.set_background_image(np.array([[[39, 39, 42]]], dtype=np.uint8))

def start_server_and_abort_event(request: gr.Request):
    """Start a viser server and register abort event."""
    server = viser.ViserServer()
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.gui.configure_theme(dark_mode=True, show_share_button=False, control_layout="collapsible")
        set_bkgd_color(client)
    print(f"Starting server {server.get_port()}")
    server_url, tunnel = setup_tunnel(
        local_host=server.get_host(),
        local_port=server.get_port(),
        share_token=secrets.token_urlsafe(32),
        share_server_address=None,
    )
    SERVERS[request.session_hash] = (server, tunnel)
    if server_url is None:
        raise gr.Error("Failed to get a viewport URL. Please check your network connection.")
    time.sleep(1)
    ABORT_EVENTS[request.session_hash] = threading.Event()
    return (
        StaticRenderer(server),
        gr.HTML(
            f'<iframe src="{server_url}" style="display: block; margin: auto; width: 100%; height: min(60vh, 600px);" frameborder="0"></iframe>',
            container=True,
        ),
        request.session_hash,
    )

def stop_server_and_abort_event(request: gr.Request):
    """Stop viser server and abort event."""
    if request.session_hash in SERVERS:
        print(f"Stopping server {request.session_hash}")
        server, tunnel = SERVERS.pop(request.session_hash)
        server.stop()
        tunnel.kill()
    if request.session_hash in ABORT_EVENTS:
        print(f"Setting abort event {request.session_hash}")
        ABORT_EVENTS[request.session_hash].set()
        time.sleep(5)
        ABORT_EVENTS.pop(request.session_hash)

def set_abort_event(request: gr.Request):
    """Set abort event for current session."""
    if request.session_hash in ABORT_EVENTS:
        print(f"Setting abort event {request.session_hash}")
        ABORT_EVENTS[request.session_hash].set()

def get_advance_examples(selection: gr.SelectData):
    """Return advanced example images."""
    index = selection.index
    return (
        gr.Gallery(ADVANCE_EXAMPLE_MAP[index][1], visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.Gallery(visible=False),
    )

   

def main(server_port: int | None = None, share: bool = True):
    """Main entry for Gradio app."""
    with gr.Blocks(js=_APP_JS) as app:
        renderer = gr.State()
        session_hash = gr.State()
        _ = get_preamble()
        app.load(
            start_server_and_abort_event,
            outputs=[renderer, gr.HTML(container=True, render=False), session_hash],
        )
        app.unload(stop_server_and_abort_event)
    app.queue(max_size=5).launch(
        share=share,
        server_port=server_port,
        show_error=True,
        allowed_paths=[WORK_DIR],
        ssr_mode=False,
    )

if __name__ == "__main__":
    tyro.cli(main)