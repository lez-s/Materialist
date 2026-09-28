#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as NF
from torchvision.utils import save_image
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import global_config
from mymodels.mlps import PosMLP
from myutils.mesh_recon import depth_file_to_mesh, rotate_mesh_around_x
from myutils.mi_plugin import MatDiffBSDF
from myutils.misc import EarlyStopping, center_crop_and_resize, linear_to_srgb, srgb_to_linear

mi.register_bsdf("MatDiffBSDF", lambda props: MatDiffBSDF(props))

# ---------------------------------------------------------------------------
# MatNet G-buffer prediction
# ---------------------------------------------------------------------------

def predict_gbuffer(image_np, output_dir):
    """Run MatNet on image_np (linear float32 HxWx3), save predicted G-buffer to output_dir.
    Returns dict with numpy arrays: albedo, roughness, metallic, normal, depth.
    """
    from huggingface_hub import hf_hub_download
    from Material_net.dpt import MaterialNet

    model_path = hf_hub_download(
        repo_id="Lez/MatNet",
        filename="matnet_weights.pth",
        repo_type="model",
    )
    matnet = MaterialNet(
        encoder="vitb", features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False, use_clstoken=False,
    )
    matnet.load_state_dict(torch.load(model_path, weights_only=True))
    matnet = matnet.cuda()

    pred = matnet.infer_image(image_np)

    os.makedirs(output_dir, exist_ok=True)
    mi.util.write_bitmap(os.path.join(output_dir, "albedoPred.exr"), pred["albedo"])
    mi.util.write_bitmap(os.path.join(output_dir, "normalPred.exr"), pred["normal"])
    mi.util.write_bitmap(os.path.join(output_dir, "roughnessPred.png"), pred["roughness"])
    mi.util.write_bitmap(os.path.join(output_dir, "metallicPred.png"), pred["metallic"])
    mi.util.write_bitmap(os.path.join(output_dir, "depthPred.exr"), pred["depth"])
    return pred


def load_or_predict_gbuffer(image_np, output_dir):
    """Load predicted G-buffer from output_dir if it exists, otherwise run MatNet."""
    pred_files = {
        "albedo":   os.path.join(output_dir, "albedoPred.exr"),
        "normal":   os.path.join(output_dir, "normalPred.exr"),
        "roughness": os.path.join(output_dir, "roughnessPred.png"),
        "metallic": os.path.join(output_dir, "metallicPred.png"),
        "depth":    os.path.join(output_dir, "depthPred.exr"),
    }
    if all(os.path.exists(p) for p in pred_files.values()):
        print("[Info] Predicted G-buffer already exists, skipping MatNet.")
        pred = {k: np.array(mi.Bitmap(p), dtype=np.float32) for k, p in pred_files.items()}
    else:
        print("[Info] Running MatNet to predict G-buffer...")
        pred = predict_gbuffer(image_np, output_dir)

    return {
        "albedo":    torch.from_numpy(pred["albedo"]).cuda().clamp(0, 1),
        "roughness": torch.from_numpy(
            pred["roughness"][..., None] if pred["roughness"].ndim == 2 else pred["roughness"][..., :1]
        ).cuda().clamp(0.07, 1),
        "metallic":  torch.from_numpy(
            pred["metallic"][..., None] if pred["metallic"].ndim == 2 else pred["metallic"][..., :1]
        ).cuda().clamp(0, 1),
        "normal":    torch.from_numpy(pred["normal"]).cuda(),
        "depth":     pred["depth"],
    }


# ---------------------------------------------------------------------------
# Mesh reconstruction
# ---------------------------------------------------------------------------

def build_or_load_mesh(depth_np, output_dir, stem, mask_path=None):
    """Load mesh from output_dir/<stem>.ply if it exists, else reconstruct from depth."""
    mesh_path = os.path.join(output_dir, f"{stem}.ply")
    if os.path.exists(mesh_path):
        print(f"[Info] Mesh already exists: {mesh_path}")
        return mesh_path

    depth = 2 * depth_np.max() - depth_np
    if mask_path and os.path.exists(mask_path):
        import imageio.v2 as imageio
        mask_img = imageio.imread(mask_path)
        if mask_img.ndim > 2:
            mask_img = mask_img[..., 0]
        mask_img = mask_img.astype(np.float32)
        if mask_img.max() > 1.0:
            mask_img /= 255.0
        mesh_mask = mask_img > 0.5
        depth[~mesh_mask] = 0
        print(f"[Info] Applied mesh mask from {mask_path}")

    mesh, _ = depth_file_to_mesh(depth, cameraMatrix=None, minAngle=6, sun3d=False, depthScale=1.0)
    mesh = rotate_mesh_around_x(mesh, 180)
    os.makedirs(output_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"[Info] Saved mesh: {mesh_path}")
    return mesh_path


# ---------------------------------------------------------------------------
# Scene / rendering
# ---------------------------------------------------------------------------

def load_mesh_scene(mesh_path, fov=35, max_depth=4):
    cam_cfg_path = os.path.join(global_config.BASE_DIR, "myutils", "default_cam.json")
    camera = mi.load_dict({
        "type": "perspective",
        "fov": float(fov),
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]
        ),
        "film": {"type": "hdrfilm", "width": 512, "height": 512},
    })
    return mi.load_dict({
        "type": "scene",
        "shape": {
            "type": "ply",
            "filename": mesh_path,
            "bsdf": {"type": "MatDiffBSDF", "cam_meta": cam_cfg_path, "use_mesh_normal": True},
        },
        "integrator": {"type": "path", "max_depth": max_depth},
        "sensor": camera,
        "emitter": {"type": "envmap", "filename": os.path.join(global_config.ENVMAP_DIR, "0.hdr")},
    })


@dr.wrap_ad(source="torch", target="drjit")
def render_with_envmap(scene, envmap, albedo, roughness, metallic, spp=64):
    params = mi.traverse(scene)
    params["emitter.data"] = envmap
    params["shape.bsdf.a"] = albedo
    params["shape.bsdf.r"] = roughness
    params["shape.bsdf.m"] = metallic
    params.update()
    return mi.render(scene, params, spp=spp, seed=np.random.randint(0, 10000))


# ---------------------------------------------------------------------------
# Envmap optimisation
# ---------------------------------------------------------------------------

def optimize_envmap(scene, mat, target_image, output_dir, spp=64, env_h=256, env_w=512,
                    epochs=2000, coordinate_type="spherical"):
    os.makedirs(os.path.join(output_dir, "best_results"), exist_ok=True)
    input_dims = 6 if coordinate_type == "spherical" else 5
    envmap_net = PosMLP(
        in_dims=input_dims,
        out_dims=3,
        dims=[256, 256, 256, 256],
        skip_connection=[1, 3],
        weight_norm=False,
        multires_view=2,
        output_type="envmap",
        color_ch=3,
        img_h=env_h,
        img_w=env_w,
        coordinate_type=coordinate_type,
    ).cuda()

    # Set fixed materials in scene
    mi_params = mi.traverse(scene)
    mi_params["shape.bsdf.a"] = mat["albedo"]
    mi_params["shape.bsdf.r"] = mat["roughness"]
    mi_params["shape.bsdf.m"] = mat["metallic"]
    mi_params.update()

    start_envmap = torch.ones(env_h, env_w, 3, device="cuda").reshape(-1, 3)
    optimizer = torch.optim.Adam(envmap_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    early_stopping = EarlyStopping(patience=100, min_delta=0.01)
    target_srgb = linear_to_srgb(target_image)

    best_loss = float("inf")
    best_envmap = None
    best_render = None
    history = []

    with tqdm(total=epochs, desc="Optimizing envmap", unit="epoch", file=sys.stdout) as pbar:
        for epoch in range(epochs):
            envmap = envmap_net(start_envmap).reshape(env_h, env_w, 3)
            rendered = render_with_envmap(scene, envmap, mat["albedo"], mat["roughness"], mat["metallic"], spp=spp)
            rendered_srgb = linear_to_srgb(rendered)
            loss_mse = NF.mse_loss(rendered_srgb, target_srgb)
            loss_l1 = NF.l1_loss(rendered_srgb, target_srgb)
            loss = loss_mse + loss_l1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if loss_mse.item() < best_loss:
                best_loss = loss_mse.item()
                best_envmap = envmap.detach().clone()
                best_render = rendered_srgb.detach().clone()

            history.append({"epoch": epoch, "loss_mse": loss_mse.item(), "loss_l1": loss_l1.item()})
            early_stopping(loss_mse.item())
            pbar.set_postfix(loss=loss.item(), loss_mse=loss_mse.item())
            pbar.update(1)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    mi.util.write_bitmap(os.path.join(output_dir, "final_envmap.hdr"), best_envmap.cpu().numpy())
    mi.util.write_bitmap(os.path.join(output_dir, "best_results", "envmap.hdr"), best_envmap.cpu().numpy())
    mi.util.write_bitmap(os.path.join(output_dir, "best_results", "rendered_img.exr"), best_render.cpu().numpy())
    save_image(
        torch.cat([target_srgb, best_render], dim=1).permute(2, 0, 1),
        os.path.join(output_dir, "comparison.png"),
    )
    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    default_img = os.path.join(global_config.BASE_DIR, "examples", "infer_envmap", "001_im.exr")
    default_out = os.path.join(global_config.BASE_DIR, "examples", "infer_envmap")
    parser = argparse.ArgumentParser(
        description="Predict G-buffer with MatNet and infer an environment map."
    )
    parser.add_argument(
        "--image_path", default=default_img,
        help="Input image (.exr for linear, .png/.jpg assumed sRGB).",
    )
    parser.add_argument(
        "--output_dir", default=default_out,
        help="Directory to save predicted G-buffer, mesh, envmap results.",
    )
    parser.add_argument("--mask_path", default=None,
                        help="Optional mesh mask PNG (white = keep depth).")
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--env_h", type=int, default=256)
    parser.add_argument("--env_w", type=int, default=512)
    parser.add_argument("--coordinate_type", choices=["spherical", "uv"], default="spherical")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = os.path.abspath(args.image_path)
    output_dir = os.path.abspath(args.output_dir)
    stem = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)

    # Load input image
    print(f"[Info] Loading image: {image_path}")
    img_bitmap = mi.Bitmap(image_path)
    image_np = np.array(img_bitmap)
    is_srgb_input = Path(image_path).suffix.lower() != ".exr"
    if is_srgb_input and np.issubdtype(image_np.dtype, np.integer):
        image_np = image_np.astype(np.float32) / np.iinfo(image_np.dtype).max
    else:
        image_np = image_np.astype(np.float32)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    image_np = center_crop_and_resize(image_np, (512, 512), return_tensor=False)
    if is_srgb_input:
        warnings.warn(
            "Input image is PNG/JPG — assuming sRGB, converting to linear.",
            UserWarning,
        )
        image_np = srgb_to_linear(image_np)

    # Save target image if not already there
    target_exr = os.path.join(output_dir, "gt_image.exr")
    if not os.path.exists(target_exr):
        mi.util.write_bitmap(target_exr, image_np)

    # Predict G-buffer (or load cached)
    mask_path = args.mask_path or os.path.join(output_dir, "mesh_mask.png")
    mat = load_or_predict_gbuffer(image_np, output_dir)

    # Reconstruct mesh (or load cached)
    mesh_path = build_or_load_mesh(mat["depth"], output_dir, stem, mask_path=mask_path)

    # Load scene
    scene = load_mesh_scene(mesh_path)

    # Load target image as tensor
    target = torch.from_numpy(image_np).cuda()

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({
            "image_path": image_path,
            "output_dir": output_dir,
            "mesh_path": mesh_path,
            "coordinate_type": args.coordinate_type,
            "env_h": args.env_h,
            "env_w": args.env_w,
            "spp": args.spp,
            "epochs": args.epochs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    best_loss = optimize_envmap(
        scene, mat, target, output_dir,
        spp=args.spp,
        env_h=args.env_h,
        env_w=args.env_w,
        epochs=args.epochs,
        coordinate_type=args.coordinate_type,
    )
    print(f"Completed with best MSE: {best_loss:.6f}")


if __name__ == "__main__":
    main()
