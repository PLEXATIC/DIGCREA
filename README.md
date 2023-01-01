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

Computation power can either be accessed using google-colab or by using a pretty powerful pc that will soon arrive. 

## Attempts
Unfortunately, there was no time yet to start attempting to create such a video. However, at the very latest when the PC arrives, attempts will definitely be made.

# Project
## Process
We started out by trying out the following technologies:
1.	StyleGAN 3
2.	Stable Diffusion
3.	StyleGAN 2
4.	Maua StyleGAN
5.	BigGAN

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