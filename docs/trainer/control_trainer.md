# Control Trainer

The Control trainer supports channel-concatenated control conditioning for models either using low-rank adapters or full-rank training. It involves adding extra input channels to the patch embedding layer (referred to as the "control injection" layer in finetrainers), to mix conditioning features into the latent stream. This architecture choice is very common and has been seen before in many models - CogVideoX-I2V, HunyuanVideo-I2V, Alibaba's Fun Control models, etc.
