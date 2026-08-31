from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn
import torch as th

class MaskableFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        extractors: dict[str, nn.Module] = {}
        embed_dim = 8

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "action_mask": continue                           # ignore mask

            if key == "connector" and isinstance(subspace, spaces.Discrete):
                extractors[key] = nn.Embedding(subspace.n, embed_dim)
                total_concat_size += embed_dim

            elif key in ["upf_location", "direction"] and isinstance(subspace, spaces.MultiBinary):
                # represents True/False so no need to extract features
                extractors[key] = nn.Identity()
                total_concat_size += subspace.shape[0]

            elif key == "datacenters" and isinstance(subspace, spaces.Box):
                extractors[key] = nn.Embedding(int(subspace.high.max()) + 1, embed_dim)                # NOTE: spaces.Box bounds are inclusive
                total_concat_size += subspace.shape[0] * subspace.shape[1] * embed_dim

            elif key == "datacenters_available_cpu" and isinstance(subspace, spaces.Box):
                # already normalised, no need to extract features
                extractors[key] = nn.Identity()
                total_concat_size += subspace.shape[0] * subspace.shape[1]

            else:
                raise NameError(f"No knowledge on how to extract features for {key=} and {subspace=}")
        super().__init__(observation_space, features_dim=total_concat_size)
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "datacenters":
                obs = obs.long()
            encoded = extractor(obs)
            encoded_flat = encoded.flatten(start_dim=1)
            encoded_tensor_list.append(encoded_flat)
        return th.cat(encoded_tensor_list, dim=1)