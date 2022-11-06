import librosa
import argparse
import numpy as np
import moviepy.editor as mpy
import random
import torch
from tqdm import tqdm
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

import numpy as np
from PIL import Image
import scipy

torch.cuda.empty_cache()

_errstr = "Mode is unknown or incompatible with input array shape."

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


#get input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--song",required=True)
parser.add_argument("--resolution", default='512')
parser.add_argument("--duration", type=int)
parser.add_argument("--pitch_sensitivity", type=int, default=220)
parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
parser.add_argument("--depth", type=float, default=1)
parser.add_argument("--classes", nargs='+', type=int)
parser.add_argument("--num_classes", type=int, default=12)
parser.add_argument("--sort_classes_by_power", type=int, default=0)
parser.add_argument("--jitter", type=float, default=0.5)
parser.add_argument("--frame_length", type=int, default=512)
parser.add_argument("--truncation", type=float, default=1)
parser.add_argument("--smooth_factor", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--use_previous_classes", type=int, default=0)
parser.add_argument("--use_previous_vectors", type=int, default=0)
parser.add_argument("--output_file", default="output.mp4")
args = parser.parse_args()


#read song
if args.song:
    song=args.song
    print('\nReading audio \n')
    y, sr = librosa.load(song)
else:
    raise ValueError("you must enter an audio file name in the --song argument")

#set model name based on resolution
model_name='biggan-deep-' + args.resolution

frame_length=args.frame_length

#set pitch sensitivity
pitch_sensitivity=(300-args.pitch_sensitivity) * 512 / frame_length

#set tempo sensitivity
tempo_sensitivity=args.tempo_sensitivity * frame_length / 512

#set depth
depth=args.depth

#set number of classes  
num_classes=args.num_classes

#set sort_classes_by_power    
sort_classes_by_power=args.sort_classes_by_power

#set jitter
jitter=args.jitter
    
#set truncation
truncation=args.truncation

#set batch size  
batch_size=args.batch_size

#set use_previous_classes
use_previous_vectors=args.use_previous_vectors

#set use_previous_vectors
use_previous_classes=args.use_previous_classes
    
#set output name
outname=args.output_file

#set smooth factor
if args.smooth_factor > 1:
    smooth_factor=int(args.smooth_factor * 512 / frame_length)
else:
    smooth_factor=args.smooth_factor

#set duration  
if args.duration:
    seconds=args.duration
    frame_lim=int(np.floor(seconds*22050/frame_length/batch_size))
else:
    frame_lim=int(np.floor(len(y)/sr*22050/frame_length/batch_size))
    
    


# Load pre-trained model - here we could take a different model
model = BigGAN.from_pretrained(model_name)

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################
########################################
########################################
########################################
########################################


#create spectrogram
spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000, hop_length=frame_length)

#get mean power at each time point
specm=np.mean(spec,axis=0)

#compute power gradient across time points
gradm=np.gradient(specm)

#set max to 1
gradm=gradm/np.max(gradm)

#set negative gradient time points to zero 
gradm = gradm.clip(min=0)
    
#normalize mean power between 0-1
specm=(specm-np.min(specm))/np.ptp(specm)

#create chromagram of pitches X time points
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=frame_length)

#sort pitches by overall power 
chromasort=np.argsort(np.mean(chroma,axis=1))[::-1]



########################################
########################################
########################################
########################################
########################################


if args.classes:
    classes=args.classes
    if len(classes) not in [12,num_classes]:
        raise ValueError("The number of classes entered in the --class argument must equal 12 or [num_classes] if specified")
    
elif args.use_previous_classes==1:
    cvs=np.load('class_vectors.npy')
    classes=list(np.where(cvs[0]>0)[0])
    
else: #select 12 random classes
    cls1000=list(range(1000))
    random.shuffle(cls1000)
    classes=cls1000[:12]
    



if sort_classes_by_power==1:

    classes=[classes[s] for s in np.argsort(chromasort[:num_classes])]



#initialize first class vector
cv1=np.zeros(1000)
for pi,p in enumerate(chromasort[:num_classes]):
    
    if num_classes < 12:
        cv1[classes[pi]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]       
    else:
        cv1[classes[p]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]

#initialize first noise vector
nv1 = truncated_noise_sample(truncation=truncation)[0]

#initialize list of class and noise vectors
class_vectors=[cv1]
noise_vectors=[nv1]

#initialize previous vectors (will be used to track the previous frame)
cvlast=cv1
nvlast=nv1


#initialize the direction of noise vector unit updates
update_dir=np.zeros(128)
for ni,n in enumerate(nv1):
    if n<0:
        update_dir[ni] = 1
    else:
        update_dir[ni] = -1


#initialize noise unit update
update_last=np.zeros(128)


########################################
########################################
########################################
########################################
########################################


#get new jitters
def new_jitters(jitter):
    jitters=np.zeros(128)
    for j in range(128):
        if random.uniform(0,1)<0.5:
            jitters[j]=1
        else:
            jitters[j]=1-jitter        
    return jitters


#get new update directions
def new_update_dir(nv2,update_dir):
    for ni,n in enumerate(nv2):                  
        if n >= 2*truncation - tempo_sensitivity:
            update_dir[ni] = -1  
                        
        elif n < -2*truncation + tempo_sensitivity:
            update_dir[ni] = 1   
    return update_dir


#smooth class vectors
def smooth(class_vectors,smooth_factor):
    
    if smooth_factor==1:
        return class_vectors
    
    class_vectors_terp=[]
    for c in range(int(np.floor(len(class_vectors)/smooth_factor)-1)):  
        ci=c*smooth_factor          
        cva=np.mean(class_vectors[int(ci):int(ci)+smooth_factor],axis=0)
        cvb=np.mean(class_vectors[int(ci)+smooth_factor:int(ci)+smooth_factor*2],axis=0)
                    
        for j in range(smooth_factor):                                 
            cvc = cva*(1-j/(smooth_factor-1)) + cvb*(j/(smooth_factor-1))                                          
            class_vectors_terp.append(cvc)
            
    return np.array(class_vectors_terp)


#normalize class vector between 0-1
def normalize_cv(cv2):
    min_class_val = min(i for i in cv2 if i != 0)
    for ci,c in enumerate(cv2):
        if c==0:
            cv2[ci]=min_class_val    
    cv2=(cv2-min_class_val)/np.ptp(cv2) 
    
    return cv2


print('\nGenerating input vectors \n')

for i in tqdm(range(len(gradm))):   
    
    #print progress
    pass

    #update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity
    if i%200==0:
        jitters=new_jitters(jitter)

    #get last noise vector
    nv1=nvlast

    #set noise vector update based on direction, sensitivity, jitter, and combination of overall power and gradient of power
    update = np.array([tempo_sensitivity for k in range(128)]) * (gradm[i]+specm[i]) * update_dir * jitters 
    
    #smooth the update with the previous update (to avoid overly sharp frame transitions)
    update=(update+update_last*3)/4
    
    #set last update
    update_last=update
        
    #update noise vector
    nv2=nv1+update

    #append to noise vectors
    noise_vectors.append(nv2)
    
    #set last noise vector
    nvlast=nv2
                   
    #update the direction of noise units
    update_dir=new_update_dir(nv2,update_dir)

    #get last class vector
    cv1=cvlast
    
    #generate new class vector
    cv2=np.zeros(1000)
    for j in range(num_classes):
        
        cv2[classes[j]] = (cvlast[classes[j]] + ((chroma[chromasort[j]][i])/(pitch_sensitivity)))/(1+(1/((pitch_sensitivity))))

    #if more than 6 classes, normalize new class vector between 0 and 1, else simply set max class val to 1
    if num_classes > 6:
        cv2=normalize_cv(cv2)
    else:
        cv2=cv2/np.max(cv2)
    
    #adjust depth    
    cv2=cv2*depth
    
    #this prevents rare bugs where all classes are the same value
    if np.std(cv2[np.where(cv2!=0)]) < 0.0000001:
        cv2[classes[0]]=cv2[classes[0]]+0.01

    #append new class vector
    class_vectors.append(cv2)
    
    #set last class vector
    cvlast=cv2


#interpolate between class vectors of bin size [smooth_factor] to smooth frames 
class_vectors=smooth(class_vectors,smooth_factor)


#check whether to use vectors from last run
if use_previous_vectors==1:   
    #load vectors from previous run
    class_vectors=np.load('class_vectors.npy')
    noise_vectors=np.load('noise_vectors.npy')
else:
    #save record of vectors for current video
    np.save('class_vectors.npy',class_vectors)
    np.save('noise_vectors.npy',noise_vectors)



########################################
########################################
########################################
########################################
########################################
    

#convert to Tensor
noise_vectors = torch.Tensor(np.array(noise_vectors))      
class_vectors = torch.Tensor(np.array(class_vectors))      


#Generate frames in batches of batch_size

print('\n\nGenerating frames \n')

#send to CUDA if running on GPU
model=model.to(device)
noise_vectors=noise_vectors.to(device)
class_vectors=class_vectors.to(device)


frames = []

for i in tqdm(range(frame_lim)):
    
    #print progress
    pass

    if (i+1)*batch_size > len(class_vectors):
        torch.cuda.empty_cache()
        break
    
    #get batch
    noise_vector=noise_vectors[i*batch_size:(i+1)*batch_size]
    class_vector=class_vectors[i*batch_size:(i+1)*batch_size]

    # Generate images
    with torch.no_grad():
        output = model(noise_vector, class_vector, truncation)

    output_cpu=output.cpu().data.numpy()

    #convert to image array and add to frames
    for out in output_cpu:    
        im=np.array(toimage(out))
        frames.append(im)
        
    #empty cuda cache
    torch.cuda.empty_cache()



#Save video  
aud = mpy.AudioFileClip(song, fps = 44100) 

if args.duration:
    aud.duration=args.duration

clip = mpy.ImageSequenceClip(frames, fps=22050/frame_length)
clip = clip.set_audio(aud)
clip.write_videofile(outname,audio_codec='aac')

torch.cuda.empty_cache()
