import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_hidream_image import MoEGate


def patch_MoEGate_forward() -> None:
    MoEGate.forward = _patched_MoEGate_forward


def _patched_MoEGate_forward(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape
    # print(bsz, seq_len, h)
    ### compute gating score
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(hidden_states, self.weight, None)
    if self.scoring_func == "softmax":
        scores = logits.softmax(dim=-1)
    else:
        raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

    ### select top-k experts
    topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

    ### norm gate to sum 1
    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator

    # Completely disable auxiliary loss for lora finetuning
    # TODO(aryan): revisit this when adding support full finetuning and enabling lora on the expert layers
    aux_loss = None

    # ### expert-level computation auxiliary loss
    # if self.training and self.alpha > 0.0:
    #     scores_for_aux = scores
    #     aux_topk = self.top_k
    #     # always compute aux loss based on the naive greedy topk method
    #     topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
    #     if self.seq_aux:
    #         scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
    #         ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
    #         ce.scatter_add_(
    #             1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
    #         ).div_(seq_len * aux_topk / self.n_routed_experts)
    #         aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
    #     else:
    #         mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
    #         ce = mask_ce.float().mean(0)

    #         Pi = scores_for_aux.mean(0)
    #         fi = ce * self.n_routed_experts
    #         aux_loss = (Pi * fi).sum() * self.alpha
    # else:
    #     aux_loss = None

    return topk_idx, topk_weight, aux_loss
