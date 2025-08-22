import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from large_spatial_model.configs import LSMConfig
from large_spatial_model.gaussian_head import GaussianHead
from large_spatial_model.lseg import LSegFeatureExtractor
from large_spatial_model.utils.points_process import merge_points

from vggt.heads.dpt_head import DPTHead
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .utils.weight_modify import checkpoint_filter_fn

from distiller.vggt.models.vggt import Distiller


class VG3R(nn.Module):

    def __init__(self, config: LSMConfig):
        super().__init__()
        self.config = LSMConfig(**config)

        _url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        weights = torch.hub.load_state_dict_from_url(_url, map_location='cpu')

        # loading VGGT ckpt as a distiller
        self.distiller = Distiller()
        self.distiller.load_state_dict(weights, strict=True)
        print("Freezing distiller")
        self.distiller.eval()
        for param in self.distiller.parameters():
            param.requires_grad = False

        self.vggt = VGGT(patch_size=16)
        weights = weights.get('state_dict', weights)
        weights = checkpoint_filter_fn(weights, self.vggt)
        self.vggt.load_state_dict(weights, strict=False)
        self.config.freeze_dust3r = False

        self.dpt_head = DPTHead(dim_in=2 * 1024, feature_only=True, input_identity=True, patch_size=16)
        self.gs_attr_proj = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 123, kernel_size=1))

        # Initialize components with typed configs
        self.gaussian_head = GaussianHead(**self.config.gaussian_head_config)
        self.lseg_feature_extractor = LSegFeatureExtractor.from_pretrained(
            **self.config.lseg_config)
        self.tokenizer = nn.AvgPool2d(kernel_size=8, stride=8)  # pool each 8*8 patch into a single value
        self.feature_expansion = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(64, 512, kernel_size=1, stride=1))  # (b, 64, h, w) -> (b, 512, h//2, w//2)
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )  # (b, 512, h//2, w//2) -> (b, d_features, h, w)
        # freeze parameters and set to eval mode
        if self.config.freeze_dust3r:
            print("Freezing vggt")
            self.vggt.eval()
            for param in self.vggt.parameters():
                # if 'camera_head' not in name:  # TODO
                param.requires_grad = False
        if self.config.freeze_lseg:
            print("Freezing lseg")
            self.lseg_feature_extractor.eval()
            for param in self.lseg_feature_extractor.parameters():
                param.requires_grad = False

        # self.load_state_dict(torch.load('../checkpoint-last.pth', map_location='cpu')['model'], strict=False)

    def forward(self, view1, view2):

        # predict depth from distiller here
        images = torch.stack((view1['img'], view2['img']), 1)
        B, V, _, H, W = images.shape
        images = images.view(-1, 3, H, W)
        images_up = F.interpolate(images, size=(518, 518), mode='bilinear', align_corners=False)
        images_up = images_up.view(B, V, 3, 518, 518)
        with torch.no_grad():
            distiller_outputs = self.distiller((images_up + 1) / 2)

        # normalize intrinsics, assuming the images width and height are same here
        view1_intr = torch.concat((view1['camera_intrinsics'][:, :2,:] / view1['img'].shape[2],
            view1['camera_intrinsics'][:, 2:, :]), dim=1)
        view2_intr = torch.concat((view2['camera_intrinsics'][:, :2,:] / view2['img'].shape[2],
            view2['camera_intrinsics'][:, 2:, :]), dim=1)
        images_intr = torch.stack((view1_intr, view2_intr), 1)

        images = torch.stack((view1['img'], view2['img']), 1)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if self.config.freeze_dust3r:
            with torch.no_grad():
                outputs = self.vggt((images + 1) / 2, images_intr)
        else:
            outputs = self.vggt((images + 1) / 2, images_intr)
        extr, intr = pose_encoding_to_extri_intri(outputs['pose_enc'], view1['img'].shape[2:])
        extr = F.pad(extr, (0, 0, 0, 1), value=0)
        extr[..., 3, 3] = 1

        feats = self.dpt_head(**outputs['tokens'], images=images)
        outputs['gs_attrs'] = self.gs_attr_proj(feats.flatten(0, 1)).reshape(
            *feats.shape[:2], -1, *feats.shape[-2:])

        # LSeg forward pass
        lseg_token_feature, lseg_res_feature = self.extract_lseg_features(view1, view2)

        # Gaussian head forward pass
        final_output = self.gaussian_head(outputs, lseg_res_feature)
        for i in range(2):
            final_output[i]['depth'] = outputs['depth'][:, i]
            final_output[i]['extr'] = extr[:, i]
            final_output[i]['intr'] = intr[:, i]
            # for depth map supervision
            final_output[i]['distiller_depth'] = distiller_outputs['depth'][:, i]
            final_output[i]['distiller_depth_conf'] = distiller_outputs['depth_conf'][:, i]
            final_output[i]['distiller_world_points'] = distiller_outputs['world_points'][:, i]
            final_output[i]['distiller_world_points_conf'] = distiller_outputs['world_points_conf'][:, i]
        return final_output

    def extract_lseg_features(self, view1, view2):
        # concat view1 and view2
        img = torch.cat([view1['img'], view2['img']], dim=0)  # (v*b, 3, h, w)

        # extract features
        lseg_features = self.lseg_feature_extractor.extract_features(img)  # (v*b, 512, h//2, w//2)
        # average pooling
        lseg_token_feature = self.tokenizer(lseg_features)
        # reshape to (b, 2v, d)
        lseg_token_feature = rearrange(lseg_token_feature, '(v b) c h w -> b (v h w) c', v=2)
        # feature reduction
        lseg_res_feature = self.feature_reduction(lseg_features)

        return lseg_token_feature, lseg_res_feature

    @classmethod
    def from_pretrained(cls,
                        checkpoint_path: str,
                        use_pretrained_lseg: bool = True,
                        use_pretrained_dust3r: bool = True,
                        device: str = 'cuda'):
        ckpt = torch.load(checkpoint_path, map_location='cpu')  # load checkpoint to cpu for saving memory
        args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
        print(f"instantiating {args}")
        model = eval(args)
        state_dict = ckpt['model']
        # if use_pretrained_lseg, remove lseg related keys
        if use_pretrained_lseg:
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith('lseg_feature_extractor')
            }
        # if use_pretrained_dust3r, remove dust3r related keys
        if use_pretrained_dust3r:
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith('dust3r')
            }
        model.load_state_dict(state_dict, strict=False)
        del ckpt
        return model.to(device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    from torch.utils.data import DataLoader
    from large_spatial_model.utils.path_manager import init_all_submodules
    init_all_submodules()

    # Add configuration file related parameters
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to configuration file')
    parser.add_argument(
        '--overrides', type=str, nargs='+', default=[],
        help='Override parameters in config file, format: key=value')

    args = parser.parse_args()

    # Load configuration from YAML file
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)

    # Process command line override parameters
    if args.overrides:
        overrides = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, overrides)

    # Convert to LSMConfig type
    config = LSMConfig(**config)

    # Initialize model
    model = LSM_Dust3R(config)
    model.to('cuda')
    # Load Data
    from large_spatial_model.datasets.scannet import Scannet
    dataset = Scannet(split='train', ROOT="data/scannet_processed", resolution=256)
    # Print dataset
    print(dataset)
    # Test model
    data_loader = DataLoader(dataset, batch_size=3, shuffle=True)
    data = next(iter(data_loader))
    # move data to cuda
    for view in data:
        view['img'] = view['img'].to('cuda')
        view['depthmap'] = view['depthmap'].to('cuda')
        view['camera_pose'] = view['camera_pose'].to('cuda')
        view['camera_intrinsics'] = view['camera_intrinsics'].to('cuda')
    # Forward pass
    output = model(*data[:2])

    # Loss
    from large_spatial_model.loss import GaussianLoss
    loss = GaussianLoss()
    loss_value = loss(*data, *output, model)
    print(loss_value)
