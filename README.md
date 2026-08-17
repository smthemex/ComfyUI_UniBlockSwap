# ComfyUI_UniBlockSwap
A universal swap node that supports ComfyUI native workflow, allowing 4_6G users to experience Minimax  or Klein9B or Bernini or other large models

# Tips
* num block 为-1，单块卸载，为0时关闭swap功能， 大于0时，缓存对应数量的block值到cuda，大于等于总block时，关闭swap（等同于0）；  
* When num block is -1, unload a single block. When it is 0, turn off the swap function. When it is greater than 0, cache the corresponding number of block values to CUDA. When it is greater than or equal to the total block, turn off the swap (equivalent to 0) ；  

# Update
* 修复minimax music 的te模式出错的问题，因为机制的修改，推理速度不一定比原生的快（只是为了方便极低显存用户）  
* Fix the issue of TE mode error in minimax music, as the inference speed may not be faster than native due to mechanism modifications (only for the convenience of extremely low memory users)   
* num block的机制改成预缓存机制，调大后，gpu会预缓存对应数值的block（修复改动num block值导致lora失效的挂载bug）；
* Change the mechanism of num block to a pre caching mechanism. After scaling up, the GPU will pre cache the corresponding block values (fix the mounting bug where changing the num block value caused lora to fail);  
* ~~fix lora unuseful's bug  修复lora失效的bug，新增节点清理TE（如果推理完第一次，再次修改提示词会导致显存占用，详看插件工作流）~~
* ~~Fix gguf loader cause high ram error,修复gguf加载时内存占用过大的bug，使用时注意避免推理过大分辨率或者时长过长，导致调用共享显存（如果调用了，就变慢了，不划算）~~
* ~~Make it for ' low Vram and normal Ram' users to esay running ComfyUI origin workflows.(Support allmot all of comfyUI origin workflows)~~
* ~~Support text encoder or diffusion models, is enable text encoder will need more Ram~~

# Installation  
----

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_UniBlockSwap
```

# Example
* minimax music text encoder
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/example_music.png)
* run minimax H3 5min 0.4 just need 4.5G Vram (要降低Te的占用需要加te模块,或者用comfyUI自带的Vbar,OOM再加TE swap,避免内存占用)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/example_minimax.png)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/minimax.png)
* run bernini int4 +loras ,512x384x120frames just need 9-10G Vram (if unpack node,notice batch size is wrong 注意官方模板解开后，batch size指向是错的，须改成1)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/bernini.png)
* run klein9B Q8 just need 4.8G Vram
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/klein9B.png)
* run boogu edit bf16 (Ram is not really used)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/boogu.png)
* run krea2  bf16 (Ram is not really used)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/krea2.png)
