# Social Network Development - Image Generation Module

## Project Overview

This task is part of a larger project where our team is developing a complete **social network application**. The application will allow users to create accounts, share posts, interact with each other, and personalize their profiles. It will include features such as user authentication, post management, and recommendations using a combination of modern web technologies (Spring for backend, React for frontend, NoSQL databases).

The overall project spans multiple phases, covering different modules that will come together to build the full social network. These include:

- User management and authentication
- Post creation and interaction
- Image generation for avatars and posts
- Recommendation systems for friends and content
- Continuous integration and deployment

### Current Focus: Image Generation Module

For this week, we will be a team of seven students, six from France and one from Anvers in Belgium. Since we come from different countries, we communicate primarily in English. 

Members of the group:
- Mattéo Cuisset
- Jean-Alexis Delcambre
- Thomas Mouton
- Loïc Van Camp
- Lucas Copin
- Théo Vanbandon
- Swann Waeles

This week, our focus is on the **image generation module**, an essential feature of the social network. Users will be able to generate personalized images, such as avatars or illustrations, using AI-powered models. This module leverages the **Stable Diffusion** model to create images from text descriptions (prompts), modify existing images, and apply personalized elements to images.

The image generation service will be integrated into the larger social network application through a web service interface, enabling seamless interaction with other parts of the app.

## Image Generation Functionality

The image generation system is based on **Stable Diffusion**, a state-of-the-art deep learning model for generating high-quality images. We have implemented the following core functionalities:

1. **Text to Image (Text2Image)**:  
   Users can enter a text prompt, and the system will generate an image that reflects the content of the description. Multiple styles (e.g., black & white, portrait, abstract) are offered for further customization.

2. **Image to Image (Image2Image)**:  
   This functionality allows users to upload an existing image and provide a prompt to transform it. The result is a modified version of the original image, using different transformation strategies.

3. **Inpainting**:  
   Inpainting enables users to edit specific parts of an image, such as removing or adding objects. The user can define the area to modify by providing a mask or letting the system generate one automatically based on the prompt.

4. **Fine-tuning with Dreambooth/LoRA**:  
   Users can provide personal photos, and the system will fine-tune the model to generate personalized images. This technique is useful for generating custom avatars or illustrations with unique objects.

5. **Inpainting with Fine-tuning**:  
   This technique combines inpainting and fine-tuning, allowing users to modify an image with personalized objects while maintaining the original image’s structure.

## Expectations and Models to Implement

The image generation system is expected to produce high-quality, customizable images that integrate smoothly with the social network. The models are based on the **Stable Diffusion** architecture and make use of various techniques like:

- **Text2Image** for creating images from user prompts.
- **Image2Image** for transforming existing images.
- **Inpainting** for editing specific regions of an image.
- **Dreambooth** and **LoRA** for fine-tuning models based on personalized content.

All these techniques are exposed through a RESTful API, allowing the social network's frontend to interact with the image generation module easily.

## How the Image Generation Works

The image generation system is built using Python. Users interact with the service through HTTP requests, sending text prompts or images. The system processes these inputs using the chosen model (Text2Image, Image2Image, etc.) and returns the generated image.

Each function (e.g., generating an image from a prompt or modifying an existing image) can be triggered through an API call. The responses are then used by the main social network application to display the images to users.

## How to launch the API

First, you need to install the dependencies of the project.<br>
They are all in the document "requirements.txt", so you can install theses with the command :

*pip install -r requirements.txt*

Before you launch the API you have to clone 2 different git repository.

First, move in "./selfhost_stablediffusion_api/generator" using the command :

*cd ./selfhost_stablediffusion_api/generator*

Now, you can clone the repositories by using followings commands :

- *git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint_v2/ ./content/checkpoints/*
- *git clone https://github.com/open-mmlab/PowerPaint.git*

Don't forget to move the models files in the right folder :

- *mv ./content/checkpoints/realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model-002.bin ./content/checkpoints/realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model.bin*
- *mv ./content/checkpoints/realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model-002.safetensors ./content/checkpoints/realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model.safetensors*

Return to the root of the projet by using *cd ../../* <br>
Finally you can launch the API with the following command :

*python -m selfhost_stablediffusion_api*

If you have some problems with the module "Inpainting", just remove the 'Inpaiting' from the tab "__all__" in the file "selfhost_stablediffusion_api\generator\__init__.py" .<br>
By doing this, features 3 and 5 won't work but the others will work

## How to test the generation of images

We purpose you a notebook for to test our API, use your own images.

**Be careful not to run the setup cells if you have already executed the above commands.**

You can find this notebook at the root of project, named "testAPI.ipynb"