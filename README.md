# Audioreactive Latent walks

## Inspiration
StyleGAN2 has fascinated us with the amazing quality of images it can generate.
A demonstration of its power can be seen here: https://thispersondoesnotexist.com

People have come up with a way to create nice videos which morph from one image to another. Because they interpolate the images in latent space this is called a "Latent Walk". A demonstration of this can be seen here (in this case landscapes are used instead of faces): https://youtu.be/R2jCpK-asEo

This is already very cool, but there's even more. People wanted to have these videos react to music, which is called "Audioreactive Latent Walk".
A demonstration of this can be seen here: https://www.youtube.com/watch?v=ffesnWqknXI (Demo starts at 5:50).

The youtuber "Nerdy Rodent" has some great tutorials on how to create such an audio-reactive latent-walk.

## Tools
There are several libraries and pretrained models that are to be used in order to create such a video. This includes the default ML-Libraries like numpy, tensorflow/pytorch, opencv, and of course a pretrained StyleGAN2 model. Potentially, the StyleGAN2 Model could be exchanged for a stable-diffusion model, as there have already been made some attempts to create videos using stable diffusion.

Computation power can either be accessed using google-colab or [lambda labs](https://lambdalabs.com/) for larger and faster training (we recommend that for the stable diffusion video generation).

## Attempts
Unfortunately, there was no time yet to start attempting to create such a video. However, at the very latest when the PC arrives, attempts will definitely be made.

# Project
## Process
We started out by trying out the following technologies:
## StyleGAN 3
Right at the beginning, we wanted to start the project with full energy and try out the latest technologies. That's why our first approach was to try out StyleGAN 3 as well. Unfortunately, we quickly found out that this requires too much computing power on the one hand, but on the other hand there are also very few examples and documentation available.
## Stable Diffusion
While one person was busy with StyleGAN 3, the other person was trying to get Stable Diffusion running. This worked fine and we were able to convert this to use audioreactive. Unfortunately, when not used on a paid GPU cloud, this took too long to tune the parameters so that we got good images and this carried through the whole video.
## StyleGAN 2
After talking to our [teacher](https://github.com/gu-ma), we looked around for an alternative in StyleGAN 2 and stumbled across this [GitHub repo](https://github.com/dvschultz/ml-art-colabs). 
## Maua StyleGAN
From the repo mentioned above, we took the [maua StyleGAN](https://github.com/dvschultz/ml-art-colabs/blob/master/maua_stylegan2_audioreactive.ipynb) approach, fixed it and played around with it. Even though it was StyleGAN it still needed a lot of computing power, which costed quite a lot to "only" run some experiments on google Colab.
## BigGAN
As we struggled with most other models, we decided to use a small to test everything with and we were even able to run BigGAN on our Laptops locally. So we went with this and found a great [Github repository](https://github.com/msieg/deep-music-visualizer), which we used to get us started.

## Final Product
Our Final Product is a mixture between BigGAN and Stable Diffusion. First, we take the music Input and create a Video with BigGAN. We take the frames of the video from BigGAN as Inputs for a Stable Diffusion Img2Img Model and create a new Video with BigGAN.

## Code Structure
1.	Cut audio in small pieces
2.	Use audio fft to get spectograms for the piece generated in step 1.
3.	Summarize the frequency from spectogram
4.	Get weighted sum of random vectors per spectogram strength (each vector has noise_dim as dimensionality)
5.	Use noise to create an image / prompt class for interpolation
6.	Use diffusion to create final image
7.	Add the images together and add the music to them, for the output video

## Parameters

## Outputs / Examples
- Check [Plexatics YouTube Channel](https://www.youtube.com/@plexatic5558/videos) for Video Examples