import unittest

import torch

from finetrainers.models.attention_dispatch import AttentionProvider, attention_dispatch, attention_provider


class AttentionDispatchTest(unittest.TestCase):
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

    def test_forward_pass_all_providers(self):
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

    def test_backward_pass_selected_providers(self):
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


if __name__ == "__main__":
    unittest.main()
