# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_projector import ProjectorConfig
from .modeling_projector import ProjectorModel

from .configuration_vllava_projector import VLLaVAQFormerConfig
from .modeling_vllava_projector import VLLaVAQFormerModel

from .configuration_ccam_projector import CCAMConfig
from .modeling_ccam_projector import CCAMModel

AutoConfig.register('projector', ProjectorConfig)
AutoModel.register(ProjectorConfig, ProjectorModel)

AutoConfig.register('vllava_qformer_projector', VLLaVAQFormerConfig)
AutoModel.register(VLLaVAQFormerConfig, VLLaVAQFormerModel)

AutoConfig.register('ccam_projector', CCAMConfig)
AutoModel.register(CCAMConfig, CCAMModel)

__all__ = ['ProjectorConfig', 'ProjectorModel', 'VLLaVAQFormerConfig', 'VLLaVAQFormerModel', 'CCAMConfig', 'CCAMModel']
