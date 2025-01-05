This directory contains the training-related specifications for all the models we support in `finetrainers`. Each model page has:

* an example training command
* inference example
* numbers on memory consumption

By default, we don't include any validation-related arguments in the example training commands. To enable validation inference, one can pass:

```diff
+ --validation_prompts "$ID_TOKEN A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions.@@@49x512x768:::$ID_TOKEN A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage@@@49x512x768" \
+ --num_validation_videos 1 \
+ --validation_steps 100
```

## Model-specific docs

* [CogVideoX](./cogvideox.md)
* [LTX-Video](./ltx_video.md)
* [HunyuanVideo](./hunyuan_video.md)