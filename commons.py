import cv2
import numpy as np


def resize_frame(image, target_height, target_width):
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_width, target_height))
    elif height < width:
        resized_image = cv2.resize(image, (target_width,
                                           int(width * target_height / height)))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,
                                      cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (int(height * target_width / width),
                                           target_height))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:
                                      resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_width, target_height))


def center_crop_frame(image, th, tw):
    h, w, c = image.shape
    x1 = int(round((h - th) / 2.))
    y1 = int(round((w - tw) / 2.))
    return image[x1:x1 + th, y1:y1 + tw, :]


def sample_frames_metafunc(stride):
    def sample_frames(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
        except:
            print('Can not open %s.' % video_path)
            pass
    
        frames = []
        frame_count = 0
    
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame = frame[:, :, ::-1]
            frames.append(frame)
            frame_count += 1
    
        indices = list(range(8, frame_count - 7, stride))
    
        frames = np.array(frames)
        frame_list = frames[indices]
        return frame_list, frame_count

    return sample_frames


def sample_clips_metafunc(stride):
    def sample_clips(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
        except:
            print('Can not open %s.' % video_path)
            pass
    
        frames = []
        frame_count = 0
    
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame = frame[:, :, ::-1]
            frames.append(frame)
            frame_count += 1
    
        indices = list(range(8, frame_count - 7, stride))

        frames = np.array(frames)
        clip_list = []
        for index in indices:
            clip_list.append(frames[index - 8: index + 8])
        clip_list = np.array(clip_list)
        return clip_list, frame_count

    return sample_clips


def preprocess_frame_metafunc(mean, std, resize_to, crop_to):
    def preprocess_frame(image):
        image = np.asarray(image, dtype=np.float64)
        image = resize_frame(image, *resize_to)
        image /= 255.
        image -= np.asarray(mean)
        image /= np.asarray(std)
        if crop_to is not None:
            image = center_crop_frame(image, *crop_to)
        return image
    
    return preprocess_frame


def preprocess_clip_metafunc(mean, std, resize_to, crop_to):
    if crop_to is not None:
        mean = [ center_crop_frame(frame, *crop_to) for frame in mean ]
        mean = np.array(mean)

    def preprocess_frame(image):
        image = np.asarray(image, dtype=np.float64)
        image = resize_frame(image, *resize_to)
        if crop_to is not None:
            image = center_crop_frame(image, *crop_to)
        return image

    def preprocess_clip(clip):
        clip = np.array([ preprocess_frame(frame) for frame in clip ])
        clip /= 255.
        clip -= mean
        clip /= np.asarray(std)
        return clip

    return preprocess_clip

