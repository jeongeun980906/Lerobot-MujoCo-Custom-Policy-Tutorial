from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
@PreTrainedConfig.register_subclass("omy_baseline")
@dataclass
class BaselineConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a var█iational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 5
    n_action_steps: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ENVIRONMENT_STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    backbone: str = 'mlp' # 'mlp' or 'transformer'
    # Vision encoder
    vision_backbone: str ="facebook/dinov3-vitb16-pretrain-lvd1689m" #"facebook/dinov2-base"
    projection_dim : int = 128
    freeze_backbone:  bool = True


    # Num hidden layers
    n_hidden_layers: int = 5
    hidden_dim: int = 512   

    ## For transformer-based architectures
    n_heads: int = 4
    dim_feedforward: int = 2048
    feedforward_activation: str = "gelu"
    dropout: float = 0.1
    pre_norm: bool = True
    n_encoder_layers: int = 6

    # Training preset
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-6

    # Learning rate scheduler parameters 
    lr_warmup_steps: int = 1000
    total_training_steps: int = 500000

    use_policy_training_preset: bool = True

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig | None:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.lr_warmup_steps,
            num_decay_steps=self.total_training_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=1e-5,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None