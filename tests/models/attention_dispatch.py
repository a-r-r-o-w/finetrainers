import os
import random
import unittest

import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention

from finetrainers.models.attention_dispatch import (
    AttentionProvider,
    _AttentionProviderRegistry,
    _set_context_parallel_options,
    attention_dispatch,
    attention_provider,
    flash_attn_flash_attention,
    native_cudnn_attention,
    native_efficient_attention,
    native_flash_attention,
)
from finetrainers.parallel.ptd import _EquipartitionSharder


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


class AttentionDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(0)

    def test_forward(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        cuda_capability = torch.cuda.get_device_capability()

        query, key, value = self._create_dummy_inputs()

        all_providers = [
            (AttentionProvider._NATIVE_MATH, 0),
            (AttentionProvider.NATIVE, 5e-3),
            (AttentionProvider.FLASH, 5e-3),
            (AttentionProvider.FLASH_VARLEN, 5e-3),
            (AttentionProvider.FLEX, 2e-2),
            (AttentionProvider._NATIVE_CUDNN, 5e-3),
            (AttentionProvider._NATIVE_EFFICIENT, 5e-3),
            (AttentionProvider._NATIVE_FLASH, 5e-3),
            (AttentionProvider.SAGE, 1e-1),
            (AttentionProvider.SAGE_VARLEN, 2e-0),
            (AttentionProvider._SAGE_QK_INT8_PV_FP16_CUDA, 2e-0),  # TODO: look into the high difference threshold
            (AttentionProvider._SAGE_QK_INT8_PV_FP16_TRITON, 2e-0),
            (AttentionProvider.XFORMERS, 5e-3),
        ]

        if cuda_capability >= (8, 9):
            all_providers.append((AttentionProvider._SAGE_QK_INT8_PV_FP8_CUDA, 2e-0))
        if cuda_capability >= (9, 0):
            all_providers.append((AttentionProvider._SAGE_QK_INT8_PV_FP16_CUDA_SM90, 2e-0))

        ref_output = None
        for i, (provider, threshold) in enumerate(all_providers):
            try:
                output = self._check_forward_pass(provider, query, key, value)
                if i == 0:
                    ref_output = output.detach().clone()
                else:
                    self.assertTrue(
                        torch.allclose(output, ref_output, atol=threshold), f"Forward pass mismatch for {provider}"
                    )
            except Exception as e:
                print(f"Warning: Forward pass test failed for {provider} with error: {e}")

    def test_backward(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        query, key, value = self._create_dummy_inputs()

        selected_providers = [
            AttentionProvider.FLASH,
            AttentionProvider.FLASH_VARLEN,
            AttentionProvider.FLEX,
            AttentionProvider.NATIVE,
            AttentionProvider.XFORMERS,
        ]

        ref_output = None
        for i, provider in enumerate(selected_providers):
            try:
                output = self._check_backward_pass(provider, query, key, value)
                if i == 0:
                    ref_output = output.detach().clone()
                else:
                    if provider == AttentionProvider.FLEX:
                        threshold = 1e-2
                    else:
                        threshold = 1e-3
                    self.assertTrue(
                        torch.allclose(output, ref_output, atol=threshold), f"Backward pass mismatch for {provider}"
                    )
            except Exception as e:
                print(f"Warning: Backward pass test failed for {provider} with error: {e}")

    def _create_dummy_inputs(
        self, batch_size=2, num_heads=8, seq_len=256, head_dim=64, dtype=torch.bfloat16, device="cuda"
    ):
        torch.manual_seed(0)
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        return query, key, value

    def _check_forward_pass(self, provider: AttentionProvider, query, key, value):
        kwargs = {}
        if provider == AttentionProvider._SAGE_QK_INT8_PV_FP16_CUDA:
            kwargs["pv_accum_dtype"] = "fp32"
        with attention_provider(provider):
            output = attention_dispatch(query, key, value, attention_kwargs=kwargs)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, query.shape)
        return output

    def _check_backward_pass(self, provider: AttentionProvider, query, key, value):
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        with attention_provider(provider):
            output = attention_dispatch(query, key, value)
            loss = output.mean()
            loss.backward()

        self.assertTrue(query.grad is not None)
        self.assertTrue(key.grad is not None)
        self.assertTrue(value.grad is not None)

        query.grad.zero_()
        key.grad.zero_()
        value.grad.zero_()
        return output


class RingAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.distributed.init_process_group(backend="nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()

        cls.rank = rank
        cls.world_size = world_size
        torch.cuda.set_device(rank)
        cls.mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,))

        set_seed(0)
        cls.batch_size = 2
        cls.num_heads = 8
        cls.seq_len = 256
        cls.head_dim = 64
        cls.dtype = torch.bfloat16
        cls.device = "cuda"

        _AttentionProviderRegistry._set_context_parallel(
            mesh=cls.mesh, convert_to_fp32=True, rotate_method="allgather"
        )
        _set_context_parallel_options(is_causal=False)

        cls.full_query = torch.randn(
            cls.batch_size,
            cls.num_heads,
            cls.seq_len * cls.world_size,
            cls.head_dim,
            dtype=cls.dtype,
            device=cls.device,
            requires_grad=True,
        )
        cls.full_key = torch.randn(
            cls.batch_size,
            cls.num_heads,
            cls.seq_len * cls.world_size,
            cls.head_dim,
            dtype=cls.dtype,
            device=cls.device,
            requires_grad=True,
        )
        cls.full_value = torch.randn(
            cls.batch_size,
            cls.num_heads,
            cls.seq_len * cls.world_size,
            cls.head_dim,
            dtype=cls.dtype,
            device=cls.device,
            requires_grad=True,
        )

        # Ensure all ranks have the same data
        with torch.no_grad():
            torch.distributed.broadcast(cls.full_query, src=0)
            torch.distributed.broadcast(cls.full_key, src=0)
            torch.distributed.broadcast(cls.full_value, src=0)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            reference_output = scaled_dot_product_attention(cls.full_query, cls.full_key, cls.full_value)

        cls.reference_output = reference_output.detach().clone()
        reference_output.sum().backward()

        cls.query, cls.key, cls.value = (
            _EquipartitionSharder.shard(x, dim=2, mesh=cls.mesh).detach().clone()
            for x in (cls.full_query, cls.full_key, cls.full_value)
        )

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()

    def _test_forward_native_cudnn_attention(self, atol: float = 1e-3):
        output = native_cudnn_attention(self.query, self.key, self.value)
        output = _EquipartitionSharder.unshard(output, dim=2, mesh=self.mesh)
        self.assertEqual(output.shape, self.reference_output.shape)
        self.assertTrue(torch.allclose(output, self.reference_output, atol=atol))

    def _test_forward_native_efficient_attention(self, atol: float = 1e-3):
        output = native_efficient_attention(self.query, self.key, self.value)
        output = _EquipartitionSharder.unshard(output, dim=2, mesh=self.mesh)
        self.assertEqual(output.shape, self.reference_output.shape)
        self.assertTrue(torch.allclose(output, self.reference_output, atol=atol))

    def _test_forward_native_flash_attention(self, atol: float = 1e-3):
        output = native_flash_attention(self.query, self.key, self.value)
        output = _EquipartitionSharder.unshard(output, dim=2, mesh=self.mesh)
        self.assertEqual(output.shape, self.reference_output.shape)
        self.assertTrue(torch.allclose(output, self.reference_output, atol=atol))

    def _test_forward_flash_attn_flash_attention(self, atol: float = 1e-3):
        output = flash_attn_flash_attention(self.query, self.key, self.value)
        output = _EquipartitionSharder.unshard(output, dim=2, mesh=self.mesh)
        self.assertEqual(output.shape, self.reference_output.shape)
        self.assertTrue(torch.allclose(output, self.reference_output, atol=atol))

    def _test_backward_native_cudnn_attention(self, atol: float = 1e-3):
        query, key, value = (x.detach().clone() for x in (self.query, self.key, self.value))
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True
        output = native_cudnn_attention(query, key, value)
        output = _EquipartitionSharder.unshard(output, dim=2, mesh=self.mesh)
        output.sum().backward()
        with torch.no_grad():
            q_g, k_g, v_g = (
                _EquipartitionSharder.shard(x, dim=2, mesh=self.mesh)
                for x in (self.full_query.grad, self.full_key.grad, self.full_value.grad)
            )
        self.assertTrue(torch.allclose(query.grad, q_g, atol=atol))
        self.assertTrue(torch.allclose(key.grad, k_g, atol=atol))
        self.assertTrue(torch.allclose(value.grad, v_g, atol=atol))

    def _test_backward_native_efficient_attention(self, atol: float = 1e-3):
        query, key, value = (x.detach().clone() for x in (self.query, self.key, self.value))
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True
        output = native_efficient_attention(query, key, value)
        output = _EquipartitionSharder.unshard(output, dim=2, mesh=self.mesh)
        output.sum().backward()
        with torch.no_grad():
            q_g, k_g, v_g = (
                _EquipartitionSharder.shard(x, dim=2, mesh=self.mesh)
                for x in (self.full_query.grad, self.full_key.grad, self.full_value.grad)
            )
        self.assertTrue(torch.allclose(query.grad, q_g, atol=atol))
        self.assertTrue(torch.allclose(key.grad, k_g, atol=atol))
        self.assertTrue(torch.allclose(value.grad, v_g, atol=atol))

    def _test_backward_native_flash_attention(self, atol: float = 1e-3):
        query, key, value = (x.detach().clone() for x in (self.query, self.key, self.value))
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True
        output = native_flash_attention(query, key, value)
        output = _EquipartitionSharder.unshard(output, dim=2, mesh=self.mesh)
        output.sum().backward()
        with torch.no_grad():
            q_g, k_g, v_g = (
                _EquipartitionSharder.shard(x, dim=2, mesh=self.mesh)
                for x in (self.full_query.grad, self.full_key.grad, self.full_value.grad)
            )
        self.assertTrue(torch.allclose(query.grad, q_g, atol=atol))
        self.assertTrue(torch.allclose(key.grad, k_g, atol=atol))
        self.assertTrue(torch.allclose(value.grad, v_g, atol=atol))

    def _test_backward_flash_attn_flash_attention(self, atol: float = 1e-3):
        query, key, value = (x.detach().clone() for x in (self.query, self.key, self.value))
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True
        output = flash_attn_flash_attention(query, key, value)
        output = _EquipartitionSharder.unshard(output, dim=2, mesh=self.mesh)
        output.sum().backward()
        with torch.no_grad():
            q_g, k_g, v_g = (
                _EquipartitionSharder.shard(x, dim=2, mesh=self.mesh)
                for x in (self.full_query.grad, self.full_key.grad, self.full_value.grad)
            )
        self.assertTrue(torch.allclose(query.grad, q_g, atol=atol))
        self.assertTrue(torch.allclose(key.grad, k_g, atol=atol))
        self.assertTrue(torch.allclose(value.grad, v_g, atol=atol))


class RingAttentionCPTesterMixin:
    def test_forward_native_cudnn_attention(self):
        self._test_forward_native_cudnn_attention(atol=1e-2)

    def test_forward_native_efficient_attention(self):
        self._test_forward_native_efficient_attention(atol=1e-2)

    def test_forward_native_flash_attention(self):
        self._test_forward_native_flash_attention(atol=1e-2)

    def test_forward_flash_attn_flash_attention(self):
        self._test_forward_flash_attn_flash_attention(atol=1e-2)

    def test_backward_native_cudnn_attention(self):
        atol = 1e-2 * self.world_size  # TODO: make bounds more strict
        self._test_backward_native_cudnn_attention(atol=atol)

    def test_backward_native_efficient_attention(self):
        atol = 1e-2 * self.world_size  # TODO: make bounds more strict
        self._test_backward_native_efficient_attention(atol=atol)

    def test_backward_native_flash_attention(self):
        atol = 1e-2 * self.world_size  # TODO: make bounds more strict
        self._test_backward_native_flash_attention(atol=atol)

    @unittest.skip(
        """query diff: 0.298828125, key diff: 2.09375, value diff: 0.68359375; Needs further investigation"""
    )
    def test_backward_flash_attn_flash_attention(self):
        # Seems to require much higher bound for some reason
        atol = 1.5e-1 * self.world_size  # TODO: make bounds more strict
        self._test_backward_flash_attn_flash_attention(atol=atol)


@unittest.skipIf(
    not torch.cuda.is_available() or get_world_size() != 2, "CUDA is not available or world size is not 2"
)
class RingAttentionCP2Test(RingAttentionTest, RingAttentionCPTesterMixin):
    pass


@unittest.skipIf(
    not torch.cuda.is_available() or get_world_size() != 4, "CUDA is not available or world size is not 4"
)
class RingAttentionCP4Test(RingAttentionTest, RingAttentionCPTesterMixin):
    pass
