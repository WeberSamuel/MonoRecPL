from typing import List, Optional
from typing_extensions import TypedDict
from torch import Tensor

class MonoRecInput(TypedDict):
    keyframe: Tensor # (B x RGB x H x W)
    keyframe_pose: Tensor # (B x 4 x 4)
    keyframe_intrinsics: Tensor # (B x 4 x 4)
    frames: List[Tensor] # (Frames x B x RGB x H x W)
    poses: List[Tensor] # (Frames x B x 4 x 4)
    intrinsics: List[Tensor] # (Frames x B x 4 x 4)
    sequence: Tensor # (B x 1)
    image_id: Tensor # (B x 1)
    stereoframe: Tensor # (B x RGB x H x W)
    stereoframe_pose: Tensor # (B x 4 x 4)
    stereoframe_intrinsics: Tensor # (B x 4 x 4)
    mvobj_mask: Tensor # (B x 1 x H x W)


class CostVolumeInput(MonoRecInput):
    cv_depths: Tensor
    inv_depth_max: Tensor
    inv_depth_min: Tensor
    cv_depth_steps: Tensor

class CostVolumeOutput(CostVolumeInput):
    cost_volume: Tensor
    single_frame_cvs: List[Tensor]
    cv_module_time: Tensor

class DepthModuleInput(CostVolumeOutput):
    image_features: List[Tensor]

class DepthModuleOutput(DepthModuleInput):
    predicted_inverse_depths: List[Tensor]

class MaskModuleInput(CostVolumeOutput):
    image_features: List[Tensor]

class MaskModuleOutput(MaskModuleInput):
    cv_mask: Tensor


class SimpleMaskModuleInput(CostVolumeOutput):
    image_features: Tensor
    predicted_inverse_depths: Tensor


class SimpleMaskModuleOutput(SimpleMaskModuleInput):
    cv_mask: Tensor

class MonoRecTrainingOutput(MaskModuleOutput):
    mono_pred: List[Tensor]
    stereo_pred: Optional[List[Tensor]]
    predicted_inverse_depths: List[Tensor]
    result: Tensor
    mask: Tensor
    target: Tensor