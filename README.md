# Audioreactive Latent walks

## Inspiration
StyleGAN2 has fascinated us with the amazing quality of images it can generate.
A demonstration of its power can be seen here: https://thispersondoesnotexist.com

People have come up with a way to create nice videos which morph from one image to another. Because they interpolate the images in latent space this is called a "Latent Walk". A demonstration of this can be seen here (in this case landscapes are used instead of faces): https://youtu.be/R2jCpK-asEo

This is already very cool, but there's even more. People wanted to have these videos react to music, which is called "Audioreactive Latent Walk".
A demonstration of this can be seen here: https://www.youtube.com/watch?v=ffesnWqknXI (Demo starts at 5:50).

The youtuber "Nerdy Rodent" has some great tutorials on how to create such an audio-reactive latent-walk.

### Tools
There are several libraries and pretrained models that are to be used in order to create such a video. This includes the default ML-Libraries like numpy, tensorflow/pytorch, opencv, and of course a pretrained StyleGAN2 model. Potentially, the StyleGAN2 Model could be exchanged for a stable-diffusion model, as there have already been made some attempts to create videos using stable diffusion.

Computation power can either be accessed using Google Colab or [lambda labs](https://lambdalabs.com/) for faster generation (we recommend that for the stable diffusion video generation).

## Project
### Process
We started by trying out the following technologies:
#### StyleGAN 3
Right at the beginning, we wanted to start the project with full energy and try out the latest technologies. That's why our first approach was to try out StyleGAN 3 as well. Unfortunately, we quickly found out that this requires too much computing power on the one hand, but on the other hand there are also very few examples and documentation available.
#### Stable Diffusion
While one person was busy with StyleGAN 3, the other person was trying to get Stable Diffusion running. This worked fine and we were able to convert this to use audioreactive. Unfortunately, when not used on a paid GPU cloud, this took too long to tune the parameters so that we got good images and this carried through the whole video.
#### StyleGAN 2
After talking to our [teacher](https://github.com/gu-ma), we looked around for an alternative in StyleGAN 2 and stumbled across this [GitHub repo](https://github.com/dvschultz/ml-art-colabs). 
##### Maua StyleGAN
From the repo mentioned above, we took the [maua StyleGAN](https://github.com/dvschultz/ml-art-colabs/blob/master/maua_stylegan2_audioreactive.ipynb) approach, fixed it and played around with it. Even though it was StyleGAN it still needed a lot of computing power, which costed quite a lot to "only" run some experiments on google Colab.



https://user-images.githubusercontent.com/73790811/210607280-051878fe-eadd-46c7-866e-ada46a95628a.mp4



https://user-images.githubusercontent.com/73790811/210607308-4a549bbb-7630-480c-9c7c-47b623f792a8.mp4


#### BigGAN
As we struggled with most other models, we decided to use a small to test everything with and we were even able to run BigGAN on our Laptops locally. So we went with this and found a great [GitHub repository](https://github.com/msieg/deep-music-visualizer), which we used to get us started.


https://user-images.githubusercontent.com/73790811/210607353-359ff2e5-3dbd-4379-8489-5a765937de14.mp4




### Final Product
Our Final Product is a mixture between BigGAN and Stable Diffusion. First, we take the music Input and create a Video with BigGAN. We take the frames of the video from BigGAN as Inputs for a Stable Diffusion Img2Img Model and create a new Video with BigGAN.




https://user-images.githubusercontent.com/73790811/210607638-172de42a-5ef7-48ef-81c8-e7eeb6c6b9fe.mp4

This video was generated with the following settings: 
truncation = 0.7
extra_detail = 0.9
max_frequency_level = 11000
low_frequency_skip = 16
frequency_band_growth_rate = 1.01
smoothing_factor = 0.1
iterations = 2
seed=42,
prompt="Photorealistic, epic, focused, sharp, cinematic lighting, 4k, 8k, octane rendering, legendary, fantasy, trippy, LSD",
num_steps=10,
unconditional_guidance_scale=7.5,
temperature=0.0,
batch_size=1,
input_image=img,
input_image_strength=0.5




https://user-images.githubusercontent.com/73790811/210607789-c834eed5-344e-4df2-8df5-d4484be12807.mp4

This one had those settings:
truncation = 0.7
extra_detail = 0.9
max_frequency_level = 11000
low_frequency_skip = 16
frequency_band_growth_rate = 1.01
smoothing_factor = 0.1
iterations = 1
seed=42,
prompt="Photorealistic, epic, focused, sharp, cinematic lighting, 4k, 8k, octane rendering, beautiful",
num_steps=15,
unconditional_guidance_scale=7.5,
temperature=0.0,
batch_size=1,
input_image=img,
input_image_strength=0.6


### Code Structure
1.	Cut audio in small pieces
2.	Use audio fft to get spectograms for the piece generated in step 1.
3.	Summarize the frequency from spectogram
4.	Get weighted sum of random vectors per spectogram strength (each vector has noise_dim as dimensionality)
5.	Use noise to create an image / prompt class for interpolation
6.	Use diffusion to create final image
7.	Add the images together and add the music to them, for the output video

### How to use
You can find the file saved as [diffused-biggan.ipynb](https://github.com/PLEXATIC/DIGCREA/blob/main/diffused_biggan.ipynb) in the root of this repository.
We recommend uploading that file onto Google Colab or [lambda labs](https://lambdalabs.com/) for faster image generation.


https://user-images.githubusercontent.com/73790811/210607940-430d3b86-f496-47a3-9c7a-2c057e9d5d50.mp4




In the file you can easily change the following parameters:
- **Input path of the music file** (we recommend using music from [Pixabay.com](https://pixabay.com/music/) as it's royalty-free)
- **Output path of the video file**

#### BigGAN specific parameters

- **[Biggan Labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)** that will be used in the images for the Video
- **Truncation** is the vector length that will be used within BigGAN
- **Extra Detail** defines how detailed the output image will be
- **Maximum Frequency Level**, from that frequency upwards the values shoudn't be considered anymore
- **Low Frequency Skip**, same as Maximum Frequency Level but for low values
- **Frequency Band Growth Rate** factor on how much to increase frequency bands
- **Smoothing Factor** defines how the random vectors will be mixed
- **Iterations** amount of times the smoothing algorithm will be applied

#### Stable Diffusion specific parameters
- **Prompt**, the prompt that will be used to generate the Stable Diffusion image
- **Number of interpolation steps**, how many steps we use to interpolate between two images
- **Number of steps to generate the image**, how many steps we use to generate the image
- **Unconditional guiding Scale**, how much the image should follow the prompt
- **Input Image strength**, how much the output differs from the input image

If the video generation takes too long, you can cancel running the second last cell, then run the last cell and it will generate a video with the images generated up to then.

### Outputs / Examples
- Check [Plexatics YouTube Channel](https://www.youtube.com/@plexatic5558/videos) for more Examples
