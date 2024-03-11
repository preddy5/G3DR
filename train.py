import argparse
import sys
import yaml
import os
import click
from shutil import copytree, ignore_patterns, rmtree
from random import random
import lpips

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms as T, utils
import math
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from src import dataset_dict
from src.unet import Unet, OSGDecoder_extended
from src.camera import compute_cam2world_matrix, sample_rays
from src.utils import TensorGroup, count_parameters

import numpy as np
import clip
from torchvision.transforms import Normalize

STYLE = 'lod_no' # Vanilla

GRAD_CLIP_MIN = 0.05
NUM_GPU = 6  # quick fix for accelerate bug in schedulers

torch.manual_seed(0)
np.random.seed(0)
def unnorm(t):
    return (t + 1) * 0.5

def linear_low2high(step, start_value, final_value, start_iter, end_iter):
    return min(final_value, start_value + ((final_value - start_value)*(step-start_iter)/(end_iter - start_iter)))

def linear_high2low(step, start_value, final_value, start_iter, end_iter):
    return max(final_value, start_value - ((start_value - final_value)*(step-start_iter)/(end_iter - start_iter)))

def save(save_dir_checkpoints, step, model, opt, scheduler):
    data = {
        'step': step,
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(data, str(save_dir_checkpoints + '/checkpoint_generic.pt'))

def tv_loss(img):
    w_variance = torch.mean(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.mean(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = (h_variance + w_variance)
    return loss

class GradientScaler(torch.autograd.Function):
  @staticmethod
  def forward(ctx, colors, sigmas, scaling):
    ctx.save_for_backward(scaling)
    return colors, sigmas, scaling

  @staticmethod
  def backward(ctx, grad_output_colors, grad_output_sigmas, grad_output_ray_dist):
    (scaling,) = ctx.saved_tensors
    scaling = scaling.clamp(GRAD_CLIP_MIN, 1)
    return grad_output_colors * scaling, grad_output_sigmas * scaling, grad_output_ray_dist

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
        output_features = torch.nn.functional.grid_sample(plane_features1, projected_coordinates, mode=mode, padding_mode=padding_mode, align_corners=False)#.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        output_features += torch.nn.functional.grid_sample(plane_features2, projected_coordinates, mode=mode, padding_mode=padding_mode, align_corners=False)#.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
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

def main(config):
    # add the path to the renderer
    sys.path.insert(0, config['G3DR']['rendering']['renderer_path'])
    from training.volumetric_rendering.renderer import ImportanceRenderer, sample_from_planes, sample_from_3dgrid, generate_planes, math_utils, project_onto_planes

    dim = 128
    image_size = 128
    accelerator = Accelerator()
    accelerator.free_memory()

    rendering_kwargs = config['G3DR']['rendering']['triplane_renderer_config']['rendering_kwargs']
    mlp_decoder_config = config['G3DR']['rendering']['triplane_renderer_config']['mlp_decoder_config']
    bs =  config['G3DR']['training']['batch_size']
    learning_rate = float(config['G3DR']['training']['learning_rate'])
    save_and_sample_every = config['G3DR']['training']['save_and_sample_every']
    results_folder = config['logging']['save_dir']
    version = config['logging']['version']
    render_3d = config['G3DR']['rendering']['render']

    dataset_class = dataset_dict[config['G3DR']['training']['dataset']]
    dataset = dataset_class(config['G3DR']['training']['dataset_folder'], image_size=image_size, config=config)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)
    unet_feature_dim = config['G3DR']['unet_feature_dim']

    w_weight, w_depth, w_clip, w_tv, w_perceptual, w_rgb = 0.0, 2, 0.35, 0.1, 2, 1 

    version = version + f'_{STYLE}_grad{GRAD_CLIP_MIN:.2f}_{image_size:.0f}_wd{w_depth:.2f}_ww{w_weight:.4f}_wp{w_perceptual:.2f}_wc{w_clip:.2f}_w_tv{w_tv:.4f}_lr{learning_rate:.6f}'
    print('Version:: ', version)

    results_folder_path = Path(results_folder)
    results_folder_path.mkdir(exist_ok = True)
    save_dir = results_folder + '/{}/{}'
    save_dir_images = save_dir.format(version, 'images')
    save_dir_checkpoints = save_dir.format(version, 'checkpoints')
    save_dir_tb = save_dir.format(version, 'tb')
    create_new= config['logging'].get('create_new', True)
    load_model = config['logging'].get('load_model', False)
    resume = False

    if accelerator.is_main_process:
        if not resume:
            if os.path.exists(save_dir.format(version, '')):
                if click.confirm('Folder exists do you want to override? '+save_dir.format(version, ''), default=True):
                    rmtree(save_dir.format(version, ''))
            if create_new:
                os.makedirs(save_dir.format(version, ''))
                os.makedirs(save_dir_checkpoints)
                os.makedirs(save_dir_images)
                os.makedirs(save_dir_tb)
                copytree('./', os.path.join(save_dir.format(version, ''), 'code'),
                    ignore=ignore_patterns('*.pyc', '.gitignore', 'tmp*', 'logs*', 'experiment_scripts*',
                                            'notebook*', '.git*', '..ipynb_checkpoints*', '*.mp4', 'model_weights*'))
    if render_3d:
        if lod_flag:
            print('Extending EG3D')
            renderer = extend_render(ImportanceRenderer, sample_from_planes, sample_from_3dgrid, project_onto_planes, math_utils)()
        else:
            renderer = ImportanceRenderer()

        eg3d_decoder = OSGDecoder_extended(unet_feature_dim, options=mlp_decoder_config)
        unet_out_dim = config['G3DR']['unet_feature_dim']*3
    else:
        renderer = None
        eg3d_decoder = None
        unet_out_dim = 3

    train_num_steps = config['G3DR']['training'].get('train_num_steps', 100000) 

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

    if not estimate_camera:
        model.rays_o = dataset.rays_o
        model.rays_d = dataset.rays_d

    if accelerator.is_main_process:
        count_parameters(model)
        print('Learning rate: ', learning_rate)
        writer = SummaryWriter(save_dir_tb)

    clip_model, _ = clip.load("ViT-B/16")
    znormalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # define losses
    perceptual_criterion = lpips.LPIPS(net='vgg')#.cuda()
    loss_fn = F.l1_loss
    step = 0

    # define optimizer
    optimizer = Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.99))
    if load_model:
        name = 'suppl_test'
        load_model = f'/../{name}/checkpoints/checkpoint_generic.pt'
        checkpoint = torch.load(load_model, map_location='cpu')
        state_dict = {k.partition('module.')[2]: checkpoint['model'][k] for k in checkpoint['model'].keys()}
        model.load_state_dict(state_dict)
        if resume:
            optimizer.load_state_dict(checkpoint['opt'])
            step = checkpoint['step']
        print('Current learning rate: ', optimizer.param_groups[0]['lr'])


    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps =(train_num_steps-step)*NUM_GPU,
                                pct_start=0.02, div_factor=25)
    print('****** num gpu *********  ', NUM_GPU)

    model, optimizer, scheduler, dataloader, clip_model, perceptual_criterion = \
        accelerator.prepare(model, optimizer, scheduler, dataloader, clip_model, perceptual_criterion)

    x_novel = None
    dataloader_iter = iter(dataloader)
    with tqdm(initial = step, total = train_num_steps, disable = not accelerator.is_main_process) as pbar:
        while step < train_num_steps:
            step += 1
            try:
                data = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                data = next(dataloader_iter)

            with torch.no_grad():
                bs = data['images'].shape[0]
                x_start = 2*data['images'] - 1
                depth_start = data['depth']
                fov_degrees = torch.ones(bs, 1).to(accelerator.device)*dataset.fov
                input_feat = torch.cat([x_start, depth_start], dim=1)
                zero = torch.tensor([0,], dtype=torch.float).to(accelerator.device)
            
            with accelerator.autocast():
                if random() > min(0.4, 2*step/train_num_steps):
                    _, _, _, planes_old = model(input_feat, return_3d_features=True, render=False)

                    rays_o = model.module.rays_o[None].repeat(bs, 1, 1).to(accelerator.device)
                    rays_d = model.module.rays_d[None].repeat(bs, 1, 1).to(accelerator.device)

                    x_canon, depth, w1, depth_ref = model.module.renderer(planes_old, model.module.decoder,
                                                    rays_o, rays_d,
                                                    model.module.rendering_options, importance_depth=data['depth'])

                    x_canon = x_canon.permute(0,2,1).reshape(-1,3,image_size,image_size)
                    depth_ref = depth_ref[:,:,:,0].permute(0,2,1).reshape(bs,-1,image_size,image_size)

                    depth = depth.permute(0,2,1).reshape(-1,1,image_size, image_size)

                    # resize to 224 if different, so that it works with clip
                    if image_size < 224:
                        gt_input_img = torch.nn.functional.interpolate(data['images'], size=[224,224], mode='bilinear')
                        x_canon_resize = torch.nn.functional.interpolate(x_canon, size=[224,224], mode='bilinear')
                    else:
                        gt_input_img = data['images']
                        x_canon_resize = x_canon

                    gt_normalized = znormalize(gt_input_img)
                    clip_embed_gt = clip_model.module.encode_image(gt_normalized)
                    x_canon_normalized = znormalize(unnorm(x_canon_resize))
                    clip_embed_x_canon = clip_model.module.encode_image(x_canon_normalized)

                    loss_fn2 = F.l1_loss
                    pred_imgs, pred_depth, gt_imgs, gt_depth = x_canon, depth, x_start, data['depth']
                    pred_clip, gt_clip = clip_embed_gt, clip_embed_x_canon
                    # 128 only
                    loss_perceptual = perceptual_criterion(x_start, x_canon)
                    loss_perceptual_log = loss_perceptual
                    loss_tv = 0
                    if step > train_num_steps//2:
                        multiply_w_clip = w_clip / 2
                        multiply_w_perceptual = w_perceptual / 2
                    else:
                        multiply_w_clip = 0
                        multiply_w_perceptual = 0
                else:
                    _, _, _, planes_old = model(input_feat, return_3d_features=True, render=False)
                    camera_params = TensorGroup(
                            angles=torch.zeros(bs,3),
                            radius=torch.ones(bs)*dataset.radius,
                            look_at=torch.zeros(bs,3),
                        )

                    start_diff = 24
                    final_diff = 6
                    start_iter = 0
                    end_iter = train_num_steps//3
                    denominator = linear_high2low(step, start_diff, final_diff, start_iter, end_iter)

                    start_clip = w_clip/16
                    multiply_w_clip = min(w_clip, start_clip + ((w_clip - start_clip)*(step-start_iter)/(end_iter - start_iter)))

                    sample_diff1 = (torch.rand(bs)*2 - 1)*np.pi/denominator
                    sample_diff2 = (torch.rand(bs)*2 - 1)*np.pi/18

                    camera_params.angles[:, 0] = np.pi/2 + sample_diff1
                    camera_params.angles[:, 1] = np.pi/2 + sample_diff2

                    cam2w = compute_cam2world_matrix(camera_params).to(accelerator.device)
                    ray_origins, ray_directions = sample_rays(cam2w, fov_degrees, [image_size,image_size])

                    x_novel, depth_novel, w1, depth_ref = model.module.renderer(planes_old, model.module.decoder,
                                                    ray_origins, ray_directions,
                                                    model.module.rendering_options)
                    depth_novel = depth_novel.permute(0,2,1).reshape(-1,1,image_size, image_size)
                    x_novel_128 = x_novel.permute(0,2,1).reshape(-1,3,image_size, image_size)

                    # resize to 224 if different, so that it works with clip
                    if image_size < 224:
                        gt_input_img = torch.nn.functional.interpolate(data['images'], size=[224,224], mode='bilinear')
                        x_novel = torch.nn.functional.interpolate(x_novel_128, size=[224,224], mode='bilinear')
                    else:
                        gt_input_img = data['images']

                    gt_normalized = znormalize(gt_input_img)
                    clip_embed_gt = clip_model.module.encode_image(gt_normalized) 
                    novel_normalized = znormalize(unnorm(x_novel))
                    clip_embed_trans = clip_model.module.encode_image(novel_normalized) 

                    loss_fn2 = F.l1_loss
                    
                    pred_imgs, pred_depth, gt_imgs, gt_depth = zero, zero, zero, zero
                    pred_clip, gt_clip = clip_embed_trans, clip_embed_gt

                    loss_tv = tv_loss(depth_novel)
                    
                    loss_perceptual_log, loss_perceptual_all = perceptual_criterion(x_start, x_novel_128, retPerLayer=True) 
                    loss_perceptual = 0
                    weights_perceptual = [0, 0, 0, 1, 1]
                    for i in range(5):
                        loss_perceptual += loss_perceptual_all[i].mean() * weights_perceptual[i]  # hack for accelerator

                    multiply_w_perceptual = w_perceptual
                    multiply_w_clip = w_clip

            loss_perceptual = loss_perceptual.mean()
            loss_rgb = loss_fn2(pred_imgs, gt_imgs).mean()
            loss_depth = loss_fn(pred_depth, gt_depth, reduction = 'mean')
            loss_clip = loss_fn(pred_clip, gt_clip, reduction = 'mean')
            loss_weight = (1 - w1).mean()
            multiply_weight = linear_low2high(step, 0, 1, 0, train_num_steps//2)

            loss_total = loss_rgb*w_rgb + loss_clip*multiply_w_clip +  w_depth*loss_depth + loss_perceptual * multiply_w_perceptual  + loss_tv * w_tv + loss_weight*multiply_weight

            # save in tensorboard
            if accelerator.is_main_process:
                if loss_rgb.item() > 0:
                    writer.add_scalar('RGB loss', loss_rgb, step)
                    writer.add_scalar('depth loss', loss_depth, step)
                writer.add_scalar('clip loss', loss_clip, step)
                writer.add_scalar('perceptual loss', loss_perceptual_log.mean(), step)
                writer.add_scalar('TV loss', loss_tv, step)
                writer.add_scalar("Total_loss", loss_total, step)

            optimizer.zero_grad()

            accelerator.backward(loss_total)
            accelerator.wait_for_everyone()
            optimizer.step()
            scheduler.step()
            accelerator.wait_for_everyone()

            # if accelerator.is_main_process:
            pbar.set_description(f'RGB: {loss_rgb:.4f}, Depth: {loss_depth:.4f}, TV: {loss_tv:.4f}, clip: {loss_clip:.4f}, Perceptual: {loss_perceptual:.4f}, loss_weight: {loss_weight:.4f}, Total: {loss_total:.4f}')
            pbar.update(1)

            if accelerator.is_main_process:
                if step != 0 and step % save_and_sample_every == 0:
                    print('version:: ', version)
                    print('Current learning rate: ', optimizer.param_groups[0]['lr'])
                    utils.save_image(data['images'], str(save_dir_images + f'/input-{step}.png'), nrow = int(math.sqrt(data['images'].shape[0])))
                    utils.save_image(depth, str(save_dir_images + f'/sample-{step}_depth.png'), nrow = int(math.sqrt(data['images'].shape[0])), normalize=True)
                    utils.save_image(unnorm(x_canon), str(save_dir_images + f'/sample-{step}.png'), nrow = int(math.sqrt(data['images'].shape[0])))

                    if type(x_novel) != type(None):
                        utils.save_image(unnorm(x_novel), str(save_dir_images + f'/sample_trans-{step}.png'), nrow = int(math.sqrt(data['images'].shape[0])))
                        utils.save_image(depth_novel, str(save_dir_images + f'/sample_trans-{step}_depth.png'), nrow = int(math.sqrt(data['images'].shape[0])), normalize=True)
                    save(save_dir_checkpoints, step, model, optimizer, scheduler)
        accelerator.print('training complete')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arch parameters')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/arch_parameters_clip.yaml')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    main(config)
