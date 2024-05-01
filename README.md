# Blog Post: Painting 3D Game Objects Using Generative AI

We can take any 3D object and essentially AI generate a skin using a 2D texture map to wrap the 3D object. 

![download](https://github.com/casblood18/Painting-3D-Game-Objects-Using-Generative-AI/assets/123738254/64ea3cd3-68f7-423e-8e97-696134f87a2e)


---
## Part 1: Rendered Images With Generated Maps
### Description
---

In this notebook, I explored loading a textured mesh from an obj file and configuring the **Renderer** with the **Rasterizer** and **Shader**, while adjusting **camera** and **light** angles. Additionally, I utilized an open-source AI image generator known as **Stable Diffusion** to generate various texture maps for wrapping the object. Then, I re-rendered the mesh with the new textures and visualized the resulting image.

![download](https://github.com/casblood18/Painting-3D-Game-Objects-Using-Generative-AI/assets/123738254/3517f00f-d705-4f01-8229-4f7ab67e2063)

### Complications
---
**Main Struggle:** My main struggle involved manipulating variables and data types to ensure compatibility. This often required extensive reference to the [**Pytorch3d documentation**](https://github.com/facebookresearch/pytorch3d/tree/main/pytorch3d) to understand how each class worked with one another and to address type mismatches when passing parameters. When trying to load in a new texture map to change the mesh's texture is when I ran into my first obstacle.

```python
im = cv2.imread("Path_to_File/new_cow_texture.png")
im = im.astype(np.float32)
im = torch.from_numpy(im/255).to(device)

mesh.textures = TexturesUV(maps=[im], faces_uvs=[faces_uvs], verts_uvs=[verts_uvs])
rendered_image = renderer(mesh)
```

**Lengthy Portion:** A smaller portion that just took time was utilizing the Stable Diffusion and ControlNet program and playing around with the values to get better generated images and to finally plot them into a grid.

![stablediffusion](https://github.com/casblood18/Painting-3D-Game-Objects-Using-Generative-AI/assets/123738254/7eaba854-5315-4a08-b80a-c5dbbf301392)

## Part 2: Update Generated Mesh
### Description
---
In this notebook, instead of using the open-source **Stable Diffusion** locally with ControlNet, I used the [**ControlNet Pipeline**](https://huggingface.co/docs/diffusers/using-diffusers/controlnet) to generate a new image utilizing the canny model to outline the rendered image of the cow obj.

![download](https://github.com/casblood18/Painting-3D-Game-Objects-Using-Generative-AI/assets/123738254/e5f78f5c-1b27-42d4-902f-d4d1daef120a)

Afterward, I formatted the image output from ControlNet and created a learnable tensor with the same shape as the original cow texture map. Next, I selected the iteration count and learning rate for the optimization loop, followed by backpropagation of the images. in each iteration, the 2 images compared are the *target_rgb* which is the tensor from the formatted ControlNet output and the *predicted_rgb* which is the new rendered image using the learning *verts_rgb* tensor as the texture map. The loss function is calculated from the difference in the tensor values which is the rgb value of each pixel in the image, from the 2 images and changes are visualized in order based on the *plot_period*.

![download](https://github.com/casblood18/Painting-3D-Game-Objects-Using-Generative-AI/assets/123738254/22d62412-755c-4ebb-baf6-1f1c89e31e0a)

### Complications
---
**Main Issue:** The main issues were similar to *Part 1 Main Issues* where I had to make sure everything was compatiable, especially formatting the ControlNet output and the rendered image using the *Renderer* to equal one another. 

```python
# Format the Image from the ControlNet output
target_rgb = transforms.ToTensor()(output)
target_rgb = target_rgb.unsqueeze(0)
target_rgb = target_rgb.permute(0, 2, 3, 1)
target_rgb = target_rgb[...,:3]

# Render the new cow changing RGBA -> RGB
images_predicted = renderer(mesh, cameras=cameras, lights=lights)
predicted_rgb = images_predicted[...,:3]
```

**Additional Issue:** A smaller compatibility issue would be formatting the rendered image to be passed through the *ControlNet Pipeline* so the output image would fit the width and height of the *target_rgb* image without having resizing and cropping issues with in the optimization step. 

```python
rendered_image = renderer(mesh)
image = rendered_image[0, ..., :3].detach().cpu().numpy()
image = image*255
image = image.astype(np.uint8)
image = image.squeeze()
image = image[:, :, :3]
```

**Comments:** A minor inconvience was overlooking the iteration amount and learning rate because I thought there was some issue, but it just needed more steps to visually see a change. Playing around and increasing the values will stabilize the loss value and predict a image very similar to the target.

## Overview
---
This project really helped me understand some of the fundamental concepts of artificial intelligence, exposure to different libraries and environments, and improving my ability to navigate and comprehend open-source documentation for effective code utilization. For the issues, just trying out different methods and slowly but surely breaking down the issues helped me become more comfortable in this environment. 
