import os
import argparse
import torch
from safetensors.torch import load_file, save_file


def convert_lora_sd(diffusers_lora_sd):
    double_block_patterns = {
        "attn.to_out.0": "img_attn.proj",
        "ff.net.0.proj": "img_mlp.0",
        "ff.net.2": "img_mlp.2",
        "attn.to_add_out": "txt_attn.proj",
        "ff_context.net.0.proj": "txt_mlp.0",
        "ff_context.net.2": "txt_mlp.2",
    }
    
    prefix = "diffusion_model."
    
    converted_lora_sd = {}
    for key in diffusers_lora_sd.keys():
        # double_blocks
        if key.startswith("transformer_blocks"):
            # img_attn
            if key.endswith("to_q.lora_A.weight"):
                # lora_A
                to_q_A = diffusers_lora_sd[key]
                to_k_A = diffusers_lora_sd[key.replace("to_q", "to_k")]
                to_v_A = diffusers_lora_sd[key.replace("to_q", "to_v")]
                
                to_qkv_A = torch.cat([to_q_A, to_k_A, to_v_A], dim=0)
                qkv_A_key = key.replace("transformer_blocks", prefix + "double_blocks").replace("attn.to_q", "img_attn.qkv")
                converted_lora_sd[qkv_A_key] = to_qkv_A
                
                # lora_B
                to_q_B = diffusers_lora_sd[key.replace("to_q.lora_A", "to_q.lora_B")]
                to_k_B = diffusers_lora_sd[key.replace("to_q.lora_A", "to_k.lora_B")]
                to_v_B = diffusers_lora_sd[key.replace("to_q.lora_A", "to_v.lora_B")]
                
                to_qkv_B = torch.block_diag(to_q_B, to_k_B, to_v_B)
                qkv_B_key = qkv_A_key.replace("lora_A", "lora_B")
                converted_lora_sd[qkv_B_key] = to_qkv_B
            
            # txt_attn
            elif key.endswith("add_q_proj.lora_A.weight"):
                # lora_A
                to_q_A = diffusers_lora_sd[key]
                to_k_A = diffusers_lora_sd[key.replace("add_q_proj", "add_k_proj")]
                to_v_A = diffusers_lora_sd[key.replace("add_q_proj", "add_v_proj")]
                
                to_qkv_A = torch.cat([to_q_A, to_k_A, to_v_A], dim=0)
                qkv_A_key = key.replace("transformer_blocks", prefix + "double_blocks").replace("attn.add_q_proj", "txt_attn.qkv")
                converted_lora_sd[qkv_A_key] = to_qkv_A
                
                # lora_B
                to_q_B = diffusers_lora_sd[key.replace("add_q_proj.lora_A", "add_q_proj.lora_B")]
                to_k_B = diffusers_lora_sd[key.replace("add_q_proj.lora_A", "add_k_proj.lora_B")]
                to_v_B = diffusers_lora_sd[key.replace("add_q_proj.lora_A", "add_v_proj.lora_B")]
                
                to_qkv_B = torch.block_diag(to_q_B, to_k_B, to_v_B)
                qkv_B_key = qkv_A_key.replace("lora_A", "lora_B")
                converted_lora_sd[qkv_B_key] = to_qkv_B
            
            # just rename
            for k, v in double_block_patterns.items():
                if k in key:
                    new_key = key.replace(k, v).replace("transformer_blocks", prefix + "double_blocks")
                    converted_lora_sd[new_key] = diffusers_lora_sd[key]
        
        # single_blocks
        elif key.startswith("single_transformer_blocks"):
            if key.endswith("to_q.lora_A.weight"):
                # lora_A
                to_q_A = diffusers_lora_sd[key]
                to_k_A = diffusers_lora_sd[key.replace("to_q", "to_k")]
                to_v_A = diffusers_lora_sd[key.replace("to_q", "to_v")]
                proj_mlp_A = diffusers_lora_sd[key.replace("attn.to_q", "proj_mlp")]
                
                linear1_A = torch.cat([to_q_A, to_k_A, to_v_A, proj_mlp_A], dim=0)
                linear1_A_key = key.replace("single_transformer_blocks", prefix + "single_blocks").replace("attn.to_q", "linear1")
                converted_lora_sd[linear1_A_key] = linear1_A
                
                # lora_B
                to_q_B = diffusers_lora_sd[key.replace("to_q.lora_A", "to_q.lora_B")]
                to_k_B = diffusers_lora_sd[key.replace("to_q.lora_A", "to_k.lora_B")]
                to_v_B = diffusers_lora_sd[key.replace("to_q.lora_A", "to_v.lora_B")]
                proj_mlp_B = diffusers_lora_sd[key.replace("attn.to_q.lora_A", "proj_mlp.lora_B")]
                
                linear1_B = torch.block_diag(to_q_B, to_k_B, to_v_B, proj_mlp_B)
                linear1_B_key = linear1_A_key.replace("lora_A", "lora_B")
                converted_lora_sd[linear1_B_key] = linear1_B
            
            elif "proj_out" in key:
                new_key = key.replace("proj_out", "linear2").replace("single_transformer_blocks", prefix + "single_blocks")
                converted_lora_sd[new_key] = diffusers_lora_sd[key]
        
        else:
            print(f"unknown or not implemented: {key}")
    
    return converted_lora_sd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_lora", type=str, required=True, help="Path to LoRA .safetensors")
    parser.add_argument("--alpha", type=float, default=None, help="Optional alpha value, defaults to rank")
    parser.add_argument("--dtype", type=str, default=None, help="Optional dtype (bfloat16, float16, float32), defaults to input dtype")
    parser.add_argument("--debug", action="store_true", help="Print converted keys instead of saving")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    converted_lora_sd = convert_lora_sd(load_file(args.input_lora))
    
    if args.alpha is not None:
        for key in list(converted_lora_sd.keys()):
            if "lora_A" in key:
                alpha_name = key.replace(".lora_A.weight", ".alpha")
                converted_lora_sd[alpha_name] = torch.tensor([args.alpha], dtype=converted_lora_sd[key].dtype)
    
    dtype = None
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    
    if dtype is not None:
        dtype_min = torch.finfo(dtype).min
        dtype_max = torch.finfo(dtype).max
        for key in converted_lora_sd.keys():
            if converted_lora_sd[key].min() < dtype_min or converted_lora_sd[key].max() > dtype_max:
                print(f"warning: {key} has values outside of {dtype} {dtype_min} {dtype_max} range")
            converted_lora_sd[key] = converted_lora_sd[key].to(dtype)
    
    if args.debug:
        for key in sorted(list(converted_lora_sd.keys())):
            print(key, converted_lora_sd[key].shape, converted_lora_sd[key].dtype)
        exit()
    
    output_path = os.path.splitext(args.input_lora)[0] + "_converted.safetensors"
    save_file(converted_lora_sd, output_path)
    print(f"saved to {output_path}")