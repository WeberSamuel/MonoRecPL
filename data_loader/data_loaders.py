from typing import Any, Optional, cast
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch
from typing import List, Tuple
from .kitti_odometry_dataset import *
from .oxford_robotcar_dataset import OxfordRobotCarDataset
from .tum_mono_vo_dataset import *
from .tum_rgbd_dataset import *
from pytorch_lightning import LightningDataModule, Trainer

class BaseDataModule(LightningDataModule):
    def __init__(
        self, dataset_type, batch_size, shuffle, validation_split, num_workers, **kwargs
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = self.dataset_type(**self.dataset_kwargs)
        len_train = int(len(self.dataset) * (1 - self.validation_split))
        len_val = len(self.dataset) - len_train
        self.train, self.val = random_split(
            self.dataset,
            [len_train, len_val],
            generator=torch.Generator().manual_seed(42),
        )        
        

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class KittiOdometryDataModule(BaseDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size=1,
        shuffle=True,
        validation_split=0.0,
        num_workers=4,
        frame_count: int = 2,
        sequences: List[str] = None,
        depth_folder: str = "image_depth",
        target_image_size: Tuple[int, int] = (256, 512),
        max_length: int = None,
        dilation: int = 1,
        offset_d: int = 0,
        use_color: bool = True,
        use_dso_poses: bool = False,
        use_color_augmentation: bool = False,
        lidar_depth: bool = False,
        dso_depth: bool = True,
        annotated_lidar: bool = True,
        return_stereo: bool = False,
        return_mvobj_mask: bool = False,
        use_index_mask: Tuple[int] = None,
    ):
        """
        Dataset implementation for KITTI Odometry.
        :param dataset_dir: Top level folder for KITTI Odometry (should contain folders sequences, poses, poses_dvso (if available)
        :param frame_count: Number of frames used per sample (excluding the keyframe). By default, the keyframe is in the middle of those frames. (Default=2)
        :param sequences: Which sequences to use. Should be tuple of strings, e.g. ("00", "01", ...)
        :param depth_folder: The folder within the sequence folder that contains the depth information (e.g. sequences/00/{depth_folder})
        :param target_image_size: Desired image size (correct processing of depths is only guaranteed for default value). (Default=(256, 512))
        :param max_length: Maximum length per sequence. Useful for splitting up sequences and testing. (Default=None)
        :param dilation: Spacing between the frames (Default 1)
        :param offset_d: Index offset for frames (offset_d=0 means keyframe is centered). (Default=0)
        :param use_color: Use color (camera 2) or greyscale (camera 0) images (default=True)
        :param use_dso_poses: Use poses provided by d(v)so instead of KITTI poses. Requires poses_dvso folder. (Default=True)
        :param use_color_augmentation: Use color jitter augmentation. The same transformation is applied to all frames in a sample. (Default=False)
        :param lidar_depth: Use depth information from (annotated) velodyne data. (Default=False)
        :param dso_depth: Use depth information from d(v)so. (Default=True)
        :param annotated_lidar: If lidar_depth=True, then this determines whether to use annotated or non-annotated depth maps. (Default=True)
        :param return_stereo: Return additional stereo frame. Only used during training. (Default=False)
        :param return_mvobj_mask: Return additional moving object mask. Only used during training. If return_mvobj_mask=2, then the mask is returned as target instead of the depthmap. (Default=False)
        :param use_index_mask: Use the listed index masks (if a sample is listed in one of the masks, it is not used). (Default=())
        """
        super().__init__(
            KittiOdometryDataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            dataset_dir=dataset_dir,
            frame_count=frame_count,
            sequences=sequences,
            depth_folder=depth_folder,
            target_image_size=target_image_size,
            max_length=max_length,
            dilation=dilation,
            offset_d=offset_d,
            use_color=use_color,
            use_dso_poses=use_dso_poses,
            use_color_augmentation=use_color_augmentation,
            lidar_depth=lidar_depth,
            dso_depth=dso_depth,
            annotated_lidar=annotated_lidar,
            return_stereo=return_stereo,
            return_mvobj_mask=return_mvobj_mask,
            use_index_mask=use_index_mask,
        )


    def __init__(self, batch_size=1, shuffle=False, validation_split=0, num_workers=4, **kwargs):
class OxfordRobotCarDataModule(BaseDataModule):
    def __init__(
        self, batch_size=1, shuffle=False, validation_split=0, num_workers=4, **kwargs
    ):

        args = {
            "sequence_folders": ["../data/oxford_robotcar/sample/stereo/centre"],
            "pose_files": ["../data/oxford_robotcar/sample/vo/vo.csv"],
            "lidar_folders": ["../data/oxford_robotcar/sample/ldmrs"],
            "model_folder": "../data/oxford_robotcar/models",
            "extrinsics_folder": "../data/oxford_robotcar/extrinsics",
            "frame_count": 2,
            "cutout": [0, 1 / 3, 0, 0],
            "scale": 0.5,
            "lidar_timestamp_range": 0.25,
        }

        args.update(kwargs)

        super().__init__(
            OxfordRobotCarDataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
        )


class TUMMonoVODataModule(BaseDataModule):
    def __init__(
        self, batch_size=1, shuffle=True, validation_split=0.0, num_workers=4, **kwargs
    ):
        super().__init__(
            TUMMonoVOMultiDataset,
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
        )


class TUMRGBDDataModule(BaseDataModule):
    def __init__(
        self, batch_size=1, shuffle=True, validation_split=0.0, num_workers=4, **kwargs
    ):
        super().__init__(
            TUMRGBDDataset,
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
        )
