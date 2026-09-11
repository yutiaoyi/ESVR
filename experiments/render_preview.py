"""Offline orthographic DVR previews; not the paper's per-primitive renderer."""
import sys
from pathlib import Path
from types import SimpleNamespace
import json
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from inference import load_ground_truth, normalized_tensor, make_global_grid, query_region
from dataset_configs import DATASET_CONFIGS
from scene import GaussianModel


def dvr(volume, axis):
    # Identical transfer function and front-to-back compositing for GT and model.
    slices = np.moveaxis(volume, axis, 0)
    rgb = np.zeros((*slices.shape[1:], 3), np.float32)
    transmittance = np.ones(slices.shape[1:], np.float32)
    for values in slices:
        intensity = np.clip(values, 0, 1)
        alpha = 1 - np.exp(-0.18 * intensity)
        color = np.stack((0.95*np.ones_like(intensity),
                          0.22 + 0.7*intensity, 0.08 + 0.7*intensity), -1)
        rgb += (transmittance * alpha)[..., None] * color
        transmittance *= 1-alpha
    return np.flipud(np.swapaxes(rgb, 0, 1)).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vorts', action='store_true')
    args = parser.parse_args()
    root = Path('output/experiments/paper_repro_p0008')
    dest = Path('output/render_previews')
    dest.mkdir(parents=True, exist_ok=True)
    runs = [('aneurysm', 'aneurysm_chunk_0', 22400),
            ('breast', 'breast_sparse_t60_chunk_0', 89600)]
    if args.vorts:
        root = Path('output/experiments/vorts0001_baseline')
        dest = root/'renders'
        dest.mkdir(parents=True, exist_ok=True)
        runs = [('vorts0001', 'vorts0001_chunk_0', 22400)]
    metadata = {'renderer': 'Offline axis-aligned DVR of voxelized PLY',
                'transfer_function': 'alpha=1-exp(-0.18*clamp(v,0,1)); RGB=(0.95,0.22+0.7*v,0.08+0.7*v)',
                'background': 'black', 'runs': []}
    for name, folder, iteration in runs:
        config = DATASET_CONFIGS[name]
        gt = normalized_tensor(load_ground_truth(config)).numpy()
        shape = gt.shape
        ply = root/folder/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
        model = GaussianModel(*shape)
        model.load_ply(str(ply))
        coords, step = make_global_grid(shape, device='cuda')
        with torch.no_grad():
            pred = query_region(model, SimpleNamespace(debug=False, compute_cov3D_python=False),
                                coords, step, ((0,0,0), shape)).clamp(0,1).cpu().numpy()
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), facecolor='#171717')
        for axis in range(3):
            images = [dvr(v, axis) for v in (gt, pred)]
            error = np.abs(images[0]-images[1]).mean(-1)
            for col, (label, img) in enumerate(zip(['GT', 'ESVR reconstruction'], images)):
                Image.fromarray(np.uint8(np.clip(img,0,1)*255)).save(dest/f'{name}_axis{axis}_{label.split()[0].lower()}.png')
                axes[axis,col].imshow(img)
                axes[axis,col].set_title(f'{label} / view {"XYZ"[axis]}', color='white')
            axes[axis,2].imshow(error, cmap='inferno', vmin=0, vmax=0.1)
            axes[axis,2].set_title('Mean RGB absolute error (0–0.1)', color='white', fontsize=9)
        for ax in axes.flat:
            ax.axis('off')
        note = name if name != 'breast' else 'Breast: inferred threshold 60/255 preprocessing'
        fig.suptitle(note+'\nOffline DVR — identical transfer function for GT and prediction', color='white')
        fig.tight_layout()
        fig.savefig(dest/f'{name}_comparison.png', dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        metadata['runs'].append({'dataset':name, 'ply':str(ply), 'gt':config['path'], 'shape_xyz':list(shape)})
        mse = float(np.mean((pred.astype(np.float64)-gt)**2))
        metadata['runs'][-1]['psnr_normalized'] = float(-10*np.log10(mse))
        if args.vorts:
            # RAW export restores source intensity units and X-fastest layout.
            ((pred*2-1).transpose(2,1,0).astype('<f4')).tofile(dest/'reconstruction_128x128x128_float32.raw')
            metadata['runs'][-1]['rmse_original_units'] = float(2*np.sqrt(mse))
            fig, axes = plt.subplots(3,3,figsize=(9,9))
            for axis in range(3):
                g=np.take(gt,64,axis=axis).T
                p=np.take(pred,64,axis=axis).T
                for col,v in enumerate([g,p,np.abs(g-p)]):
                    axes[axis,col].imshow(v,origin='lower',vmin=0,vmax=0.1 if col==2 else 1,cmap='inferno' if col==2 else 'viridis')
                    axes[axis,col].set_title(f'{["GT","ESVR","Abs error (0–0.1)"][col]} / {"XYZ"[axis]}=64')
                    axes[axis,col].axis('off')
            fig.tight_layout()
            fig.savefig(dest/'vorts0001_slices.png',dpi=140)
            plt.close(fig)
        print('Saved', dest/f'{name}_comparison.png', flush=True)
        del model
        torch.cuda.empty_cache()
    (dest/'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
