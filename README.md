# ComfyUI_UniBlockSwap
A universal swap node that supports ComfyUI native workflow, allowing 4_6G users to experience Klein9B or Bernini or other large models

# Update
* Make it for ' low Vram and normal Ram' users to esay running ComfyUI origin workflows.(Support allmot all of comfyUI origin workflows)
* Support text encoder or diffusion models, is enable text encoder will need more Ram 

# Example
* run bernini int4 +loras ,512x384x120frames just need 9-10G Vram (if unpack node,notice batch size is wrong 注意官方模板解开后，batch size指向是错的，须改成1)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/bernini.png)
* run klein9B Q8 just need 4.8G Vram
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/klein9B.png)
* run boogu edit bf16 (Ram is not really used)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/boogu.png)
* run krea2  bf16 (Ram is not really used)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/krea2.png)
