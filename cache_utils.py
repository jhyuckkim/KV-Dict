from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time ## for debug

import torch

from transformers.cache_utils import DynamicCache, CacheConfig

from kvarena.methods.KVDict.omp import omp

@dataclass
class KVDictCacheConfig(CacheConfig):
    """
    Configuration class for KV-Dict cache settings.

    Attributes:

    """

    def __init__(
        self,
        max_sparsity: Optional[int] = 14,
        error_threshold: Optional = None,
        buffer_length: Optional[int] = 128,
        approximation_length: Optional[int] = 32,
    ):
        self.max_sparsity = max_sparsity
        self.error_threshold = error_threshold
        self.buffer_length = buffer_length
        self.approximation_length = approximation_length
    
    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general
        if self.buffer_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="residual_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )
        if self.approximation_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="approximation_length",
                    correct_value="a positive integer",
                    found_value=self.approximation_length,
                ),
            )
        if self.approximation_length > self.buffer_length:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="approximation_length",
                    correct_value="larger than the buffer length",
                    found_value=self.approximation_length,
                ),
            )


class KVDictCache(DynamicCache):
    """
    Cache class for KV-Dict

    
    """

    def __init__(self, cache_config: KVDictCacheConfig) -> None:
        super().__init__()
        self._key_cache_crow_indices: List[torch.Tensor] = []
        self._key_cache_col_indices: List[torch.Tensor] = []
        self._key_cache_values: List[torch.Tensor] = []
        self._value_cache_crow_indices: List[torch.Tensor] = []
        self._value_cache_col_indices: List[torch.Tensor] = []
        self._value_cache_values: List[torch.Tensor] = []

        self.max_sparsity = cache_config.max_sparsity
        self.error_threshold = cache_config.error_threshold
        self.buffer_length = cache_config.buffer_length
        self.approximation_length = cache_config.approximation_length
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_dictionary: torch.Tensor,
        value_dictionary: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        if len(self.key_cache) < layer_idx:
            raise ValueError
        elif len(self.key_cache) == layer_idx:
            if key_states.shape[-2] > self.buffer_length:
                keys_crow_indices, keys_col_indices, keys_values = self._compress(key_states[:, :, :-self.buffer_length, :], key_dictionary)
                values_crow_indices, values_col_indices, values_values = self._compress(value_states[:, :, :-self.buffer_length, :], value_dictionary)
                self._key_cache_crow_indices.append(keys_crow_indices)
                self._key_cache_col_indices.append(keys_col_indices)
                self._key_cache_values.append(keys_values)
                self._value_cache_crow_indices.append(values_crow_indices)
                self._value_cache_col_indices.append(values_col_indices)
                self._value_cache_values.append(values_values)
            else:
                self._key_cache_crow_indices.append(torch.tensor([0], dtype=torch.int32, device=key_states.device))
                self._key_cache_col_indices.append(torch.empty(0, device=key_states.device))
                self._key_cache_values.append(torch.empty(0, device=key_states.device))
                self._value_cache_crow_indices.append(torch.tensor([0], dtype=torch.int32, device=value_states.device))
                self._value_cache_col_indices.append(torch.empty(0, device=value_states.device))
                self._value_cache_values.append(torch.empty(0, device=value_states.device))

            self.key_cache.append(key_states[:, :, -self.buffer_length:, :])
            self.value_cache.append(value_states[:, :, -self.buffer_length:, :])
            return None, key_states, None, value_states
        else:
            if len(self._key_cache_crow_indices[layer_idx]) > 1:
                compressed_key_states_to_return = torch.sparse_csr_tensor(
                    self._key_cache_crow_indices[layer_idx].to(torch.int32),
                    self._key_cache_col_indices[layer_idx].to(torch.int32),
                    self._key_cache_values[layer_idx].to(torch.float16),
                    size=(self._key_cache_crow_indices[layer_idx].shape[0]-1, key_dictionary.shape[1])
                )
            else:
                compressed_key_states_to_return = None
            if len(self._value_cache_crow_indices[layer_idx]) > 1:
                compressed_value_states_to_return = torch.sparse_csr_tensor(
                    self._value_cache_crow_indices[layer_idx].to(torch.int32),
                    self._value_cache_col_indices[layer_idx].to(torch.int32),
                    self._value_cache_values[layer_idx].to(torch.float16),
                    size=(self._value_cache_crow_indices[layer_idx].shape[0]-1, value_dictionary.shape[1])
                )
            else:
                compressed_value_states_to_return = None
            
            keys_to_return = [self.key_cache[layer_idx], key_states]
            values_to_return = [self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)

            if self.key_cache[layer_idx].shape[-2] >= self.buffer_length:
                keys_crow_indices, keys_col_indices, keys_values = self._compress(self.key_cache[layer_idx][:, :, :self.approximation_length], key_dictionary)
                values_crow_indices, values_col_indices, values_values = self._compress(self.value_cache[layer_idx][:, :, :self.approximation_length], value_dictionary)
                
                # ## for now we concat in dense states
                # compressed_key_states = torch.sparse_csr_tensor(keys_crow_indices.to(torch.int32), keys_col_indices.to(torch.int32), keys_values.to(torch.float16), size=(keys_crow_indices.shape[0]-1, key_dictionary.shape[1]))
                # compressed_value_states = torch.sparse_csr_tensor(values_crow_indices.to(torch.int32), values_col_indices.to(torch.int32), values_values.to(torch.float16), size=(values_crow_indices.shape[0]-1, value_dictionary.shape[1]))

                # compressed_key_cache = torch.sparse_csr_tensor(self._key_cache_crow_indices[layer_idx].to(torch.int32), self._key_cache_col_indices[layer_idx].to(torch.int32), self._key_cache_values[layer_idx].to(torch.float16), size=(self._key_cache_crow_indices[layer_idx].shape[0]-1, key_dictionary.shape[1]))
                # compressed_value_cache = torch.sparse_csr_tensor(self._value_cache_crow_indices[layer_idx].to(torch.int32), self._value_cache_col_indices[layer_idx].to(torch.int32), self._value_cache_values[layer_idx].to(torch.float16), size=(self._value_cache_crow_indices[layer_idx].shape[0]-1, value_dictionary.shape[1]))

                # print("shape of old compressed_key_cache", compressed_key_cache.shape)
                # print(self._key_cache_crow_indices[layer_idx].shape)
                # print(self._key_cache_col_indices[layer_idx].shape)
                # print(self._key_cache_values[layer_idx].shape)

                # print("shape of adding compressed_key_states", compressed_key_states.shape)
                # print(keys_crow_indices.shape)
                # print(keys_col_indices.shape)
                # print(keys_values.shape)

                # # concat them
                # print("shape of compressed_key_cache before to dense", compressed_key_cache.shape)
                # dense_compressed_key_cache = compressed_key_cache.to_dense()
                # print("shape of compressed_key_cache after to dense", dense_compressed_key_cache.shape)
                # print("shape of compressed_key_states before to dense", compressed_key_states.shape)
                # dense_compressed_key_states = compressed_key_states.to_dense()
                # print("shape of compressed_key_states after to dense", dense_compressed_key_states.shape)
                # compressed_key_cache = torch.cat([dense_compressed_key_cache, dense_compressed_key_states], dim=0)
                # print("shape of compressed_key_cache after concat", compressed_key_cache.shape)
                # compressed_key_cache = compressed_key_cache.to_sparse_csr()
                # print("shape of compressed_key_cache after concat and after to sparse", compressed_key_cache.shape)

                # dense_compressed_value_cache = compressed_value_cache.to_dense()
                # dense_compressed_value_states = compressed_value_states.to_dense()
                # compressed_value_cache = torch.cat([dense_compressed_value_cache, dense_compressed_value_states], dim=0)
                # compressed_value_cache = compressed_value_cache.to_sparse_csr()

                # print("shape of new compressed_key_cache", compressed_key_cache.shape)
                # print(compressed_key_cache.crow_indices().shape)
                # print(compressed_key_cache.col_indices().shape)
                # print(compressed_key_cache.values().shape)

                # self._key_cache_crow_indices[layer_idx] = compressed_key_cache.crow_indices().to(torch.int16)
                # self._key_cache_col_indices[layer_idx] = compressed_key_cache.col_indices().to(torch.int16)
                # self._value_cache_values[layer_idx] = compressed_value_cache.values().to(torch.float8_e4m3fn)
                # self._value_cache_crow_indices[layer_idx] = compressed_value_cache.crow_indices().to(torch.int16)
                # self._value_cache_col_indices[layer_idx] = compressed_value_cache.col_indices().to(torch.int16)
                # self._value_cache_values[layer_idx] = compressed_value_cache.values().to(torch.float8_e4m3fn)
                # print("***** max and min of both indices *****")
                # print(max(self._key_cache_crow_indices[layer_idx]), min(self._key_cache_crow_indices[layer_idx]))
                # print(max(keys_crow_indices), min(keys_crow_indices))
                self._key_cache_crow_indices[layer_idx] = torch.cat([self._key_cache_crow_indices[layer_idx], keys_crow_indices[1:] + self._key_cache_crow_indices[layer_idx][-1]])
                self._key_cache_col_indices[layer_idx] = torch.cat([self._key_cache_col_indices[layer_idx], keys_col_indices])
                self._key_cache_values[layer_idx] = torch.cat([self._key_cache_values[layer_idx].to(torch.float16), keys_values.to(torch.float16)]).to(torch.float8_e4m3fn)
                self._value_cache_crow_indices[layer_idx] = torch.cat([self._value_cache_crow_indices[layer_idx], values_crow_indices[1:] + self._value_cache_crow_indices[layer_idx][-1]])
                self._value_cache_col_indices[layer_idx] = torch.cat([self._value_cache_col_indices[layer_idx], values_col_indices])
                self._value_cache_values[layer_idx] = torch.cat([self._value_cache_values[layer_idx].to(torch.float16), values_values.to(torch.float16)]).to(torch.float8_e4m3fn)

                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx][:, :, self.approximation_length:], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx][:, :, self.approximation_length:], value_states], dim=-2)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            return compressed_key_states_to_return, keys_to_return, compressed_value_states_to_return, values_to_return
        
    def get_seq_length(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self._seen_tokens
    
    def _compress(self, tensor, dictionary):
        # print("compressing a tensor of shape", tensor.shape)
        head_dim = tensor.shape[-1]
        # start_time = time.time()
        # print("shape of tensor", tensor.shape)
        # print("shape of dictionary", dictionary.shape)
        # print("what is going into omp", tensor.permute(2, 0, 1, 3).reshape(1, -1, head_dim).shape)
        reshaped_tensor = tensor.permute(2, 0, 1, 3).reshape(1, -1, head_dim)
        indptr, indices, data = omp(dictionary.unsqueeze(0), reshaped_tensor, self.max_sparsity, self.error_threshold)
        # compressed_tensor = torch.sparse_csr_tensor(indptr, indices, data, size=(reshaped_tensor.shape[1], dictionary.shape[1]))
        # end_time = time.time()
        # print(f"Time to compress a single layer: {end_time - start_time:.6f} seconds")
        # print("shape of compressed_tensor", compressed_tensor.shape)
        # compressed_tensor = compressed_tensor.view(tensor.shape[0], tensor.shape[1], tensor.shape[2], -1)
        # print("shape of reshaped compressed_tensor", compressed_tensor.shape)
        # print("sparsity", self.max_sparsity)
        # print("shape of indptr", indptr.shape)
        # print("shape of indices", indices.shape)
        # print("shape of data", data.shape)
        return indptr, indices, data 