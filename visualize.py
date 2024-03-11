import argparse
import sys
import yaml
import os
import click
from shutil import copytree, ignore_patterns, rmtree
from random import random

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as T, utils
import math
from pathlib import Path

from src import dataset_dict
from src.unet import Unet, OSGDecoder_extended
from src.camera import compute_cam2world_matrix, sample_rays
from src.utils import TensorGroup, count_parameters, colorize, sample_front_circle

import imageio
import numpy as np
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

STYLE = 'lod_no' # Vanilla

torch.manual_seed(0)
np.random.seed(0)
def unnorm(t):
    return (t + 1) * 0.5

class extend_MipRayMarcher(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, colors, densities, depths, rendering_options, grad_scaling=None):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2


        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)

        if type(grad_scaling) != type(None):
            alpha, colors_mid, grad_scaling = GradientScaler.apply(alpha, colors_mid, grad_scaling)

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / (weight_total +  + 0.001)

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights, weight_total


def extend_render(ImportanceRenderer, sample_from_planes, sample_from_3dgrid, project_onto_planes, math_utils):
    def sample_from_planes_hie(plane_axes, plane_features1, plane_features2, plane_features3, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
        assert padding_mode == 'zeros'
        _, M, _ = coordinates.shape
        N, n_planes, C, H, W = plane_features1.shape
        plane_features1 = plane_features1.view(N*n_planes, C, H, W)
        N, n_planes, C, H, W = plane_features2.shape
        plane_features2 = plane_features2.view(N*n_planes, C, H, W)
        N, n_planes, C, H, W = plane_features3.shape
        plane_features3 = plane_features3.view(N*n_planes, C, H, W)

        coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
        with torch.no_grad():
            projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1).float()
        output_features = torch.nn.functional.grid_sample(plane_features1, projected_coordinates, mode=mode, padding_mode=padding_mode, align_corners=False)
        output_features += torch.nn.functional.grid_sample(plane_features2, projected_coordinates, mode=mode, padding_mode=padding_mode, align_corners=False)
        output_features += torch.nn.functional.grid_sample(plane_features3, projected_coordinates, mode=mode, padding_mode=padding_mode, align_corners=False)
        output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        return output_features

    class ImportanceRenderer_extended(ImportanceRenderer):
        def __init__(self, *args, **kwargs ):
            super().__init__(*args, **kwargs)
            self.ray_marcher = extend_MipRayMarcher()
            self.STYLE = STYLE
            self.gauss_pdf = lambda x, mean, std: 1.25*torch.exp(- ((x-mean)**2) / std)

        def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options, importance_depth=None):
            with torch.no_grad():
                self.plane_axes = self.plane_axes.to(planes.device)

                if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
                    ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
                    is_ray_valid = ray_end > ray_start
                    if torch.any(is_ray_valid).item():
                        ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                        ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
                    depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
                else:
                    # Create stratified depth samples
                    depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

                batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

                # Coarse Pass
                sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
                sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_coarse = out['rgb']
            densities_coarse = out['sigma']
            colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
            densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

            # Fine Pass
            N_importance = rendering_options['depth_resolution_importance']
            if N_importance > 0:
                depths_fine = None
                with torch.no_grad():
                    if type(importance_depth) != type(None):
                        bs = importance_depth.shape[0]
                        importance_depth_reshape = importance_depth.permute(0,2,3,1).reshape(bs, -1, 1)
                        importance_depth_reshape = importance_depth_reshape[:,:,None,:]
                        if random() > 0.6:
                            var = 0.05
                            sample_uniform = torch.linspace(-var, var, N_importance).to(planes.device)
                            depths_fine = importance_depth_reshape + sample_uniform[None,None,:,None]
                    if depths_fine is None:
                        _, _, weights, weight_total = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
                        depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

                    sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
                    sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine 
                                        * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

                out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
                colors_fine = out['rgb']
                densities_fine = out['sigma']
                colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
                densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

                all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                    depths_fine, colors_fine, densities_fine)

                # Aggregate
                output_depth = all_depths
                depths_mid = (output_depth[:, :, :-1] + output_depth[:, :, 1:]) / 2

                if type(importance_depth) != type(None):
                    grad_scaling = self.gauss_pdf(depths_mid, importance_depth_reshape, 0.03)
                else:
                    grad_scaling = None
                rgb_final, depth_final, weights, weight_total = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options, grad_scaling)
            else:
                output_depth = depths_coarse
                rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
            return rgb_final, depth_final, weight_total, depths_mid

        def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
            if self.STYLE == 'lod_no':
                planes_reshape = planes.view(planes.shape[0], -1, planes.shape[-2], planes.shape[-1])

                planes_64 = F.interpolate(planes_reshape, scale_factor=0.5, mode="bilinear", antialias=True)
                planes_64 = planes_64.view(planes.shape[0], 3, -1, planes.shape[-2]//2, planes.shape[-1]//2)

                planes_32 = F.interpolate(planes_reshape, scale_factor=0.25, mode="bilinear", antialias=True)
                planes_32 = planes_32.view(planes.shape[0], 3, -1, planes.shape[-2]//4, planes.shape[-1]//4)

                planes_128 = planes
                
                sampled_features = sample_from_planes_hie(self.plane_axes, planes_128, planes_64, planes_32, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
                sampled_features = sampled_features.mean(1, keepdims=True)
            else:
                sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
                sampled_features = sampled_features.mean(1, keepdims=True)
  

            coordinates = (2/options['box_warp']) * sample_coordinates
            out = decoder(sampled_features, sample_directions)
            if options.get('density_noise', 0) > 0:
                out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
            return out
    return ImportanceRenderer_extended

def generate_video(unet, planes, num_frames=128, vimg_size=128, output_path='./output/', filename=0):
    device = planes.device
    bs = planes.shape[0]
    camera_params = TensorGroup(
        angles=torch.zeros(1,3),
        fov=torch.ones(1)*18,
        radius=torch.ones(1)*5,
        look_at=torch.zeros(1,3),
    )
    camera_params.angles[:, 0] = camera_params.angles[:, 0]+np.pi/2
    camera_params.angles[:, 1] = camera_params.angles[:, 1]+np.pi/2

    camera_samples = sample_front_circle(camera_params, num_frames)
    cam2w = compute_cam2world_matrix(camera_samples)

    ray_origins, ray_directions = sample_rays(cam2w, camera_samples.fov[:, None], [vimg_size,vimg_size])

    frames = []
    frames_depth = []
    print('Visualizing file: ', filename)
    for th in tqdm(range(num_frames)):

        rays_o, rays_d = ray_origins[th].to(device), ray_directions[th].to(device)
        rays_o, rays_d = rays_o[None].repeat(bs, 1, 1), rays_d[None].repeat(bs, 1, 1)

        rgb_out, depth, _, _ = unet.renderer(planes, unet.decoder, rays_o, rays_d, unet.rendering_options)
        if False: # choose your upsampling network
            rgb_out = rgb_out.reshape(bs, vimg_size, vimg_size, 3).permute(0,3,1,2)
            rgb_out = (rgb_out+1)* 0.5
            if False: # choose your upsampling network
                rgb_reshape = unet.model_sr_rcan(rgb_out).clamp(0, 1).cpu()
            else:
                rgb_reshape = model_sr(rgb_out).clamp(0, 1).cpu()
        else:
            rgb = (rgb_out+1)* 0.5
            rgb_reshape = rgb.reshape(bs, vimg_size, vimg_size, 3).permute(0,3,1,2).cpu()
        depth_reshape = depth.reshape(bs, vimg_size, vimg_size, 1).permute(0,3,1,2).cpu()
        depth_reshape = colorize(depth_reshape, cmap='magma_r')
        depth_reshape = torch.from_numpy(depth_reshape).to(rgb_reshape.device).permute(0,3,1,2)[:,:3]/255
        
        combined = torch.cat([rgb_reshape, depth_reshape], dim=3)
        combined = make_grid(combined, nrow = int(math.sqrt(bs)))
        frames.append((255*np.clip(combined.permute(1,2,0).cpu().detach().numpy(), 0, 1)).astype(np.uint8))

    imageio.mimwrite(os.path.join(output_path, f'output-{filename}.mp4'), frames, fps=40, quality=8)

    
def main(config, args):
    # add the path to the renderer
    sys.path.insert(0, config['G3DR']['rendering']['renderer_path'])
    print("Eg3D folder: ", config['G3DR']['rendering']['renderer_path'])
    # importing necessary function from Eg3D
    from training.volumetric_rendering.renderer import ImportanceRenderer, sample_from_planes, sample_from_3dgrid, generate_planes, math_utils, project_onto_planes

    dim = 128
    image_size = 128

    rendering_kwargs = config['G3DR']['rendering']['triplane_renderer_config']['rendering_kwargs']
    mlp_decoder_config = config['G3DR']['rendering']['triplane_renderer_config']['mlp_decoder_config']
    bs =  config['G3DR']['training']['batch_size']
    learning_rate = float(config['G3DR']['training']['learning_rate'])
    save_and_sample_every = config['G3DR']['training']['save_and_sample_every']
    results_folder = config['logging']['save_dir']
    version = config['logging']['version']
    render_3d = config['G3DR']['rendering']['render']

    dataset_class = dataset_dict[config['G3DR']['training']['dataset']]
    dataset = dataset_class(args.folder, image_size=image_size, config=config)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=8)
    unet_feature_dim = config['G3DR']['unet_feature_dim']

    if STYLE!='vanilla':
        print('Extending EG3D')
        renderer = extend_render(ImportanceRenderer, sample_from_planes, sample_from_3dgrid, project_onto_planes, math_utils)()
    else:
        renderer = ImportanceRenderer()
    eg3d_decoder = OSGDecoder_extended(unet_feature_dim, options=mlp_decoder_config)
    unet_out_dim = config['G3DR']['unet_feature_dim']*3

    model = Unet(
        channels = 4,
        dim = dim,
        out_dim = unet_out_dim,
        renderer=renderer,
        eg3d_decoder=eg3d_decoder,
        rendering_options=rendering_kwargs,
        dim_mults = (1, 2, 4, 8),
        render_3d=render_3d,
        image_size = image_size,
        config = config
    )
    model.rays_o = dataset.rays_o
    model.rays_d = dataset.rays_d
    
    if args.load_model!='default.pt':
        load_model = args.load_model
        checkpoint = torch.load(load_model, map_location='cpu')
        state_dict = {k.partition('module.')[2]: checkpoint['model'][k] for k in checkpoint['model'].keys()}
        model.load_state_dict(state_dict)
        model.eval()
    else:
        raise Exception("Please provide pretrained model file")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataloader_iter = iter(dataloader)
    step = 0
    
    for data in  dataloader_iter:
        with torch.no_grad():
            step += 1
            bs = data['images'].shape[0]
            x_start = 2*data['images'] - 1
            depth_start = data['depth']
            input_feat = torch.cat([x_start, depth_start], dim=1).to(device)

            _, _, _, planes = model(input_feat, return_3d_features=True, render=False)

            generate_video(model, planes, output_path=args.output_path, filename=step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arch parameters')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/arch_parameters_clip.yaml')
    parser.add_argument('--load_model',
                        dest="load_model",
                        help =  'model.pt file local',
                        default='default.pt')
    parser.add_argument('--folder',
                        dest="folder",
                        help =  'folder with rgbd images',
                        default='./images/1/')
    parser.add_argument('--output_path',
                        dest="output_path",
                        help =  'folder to output visualization video',
                        default='./output/')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    main(config, args)
