# Make sure the frame only has 1 channel data, if not, take only the first channel
def single_channel(frame):
    if frame.ndim == 4:  # Multi-channel case (Z, C, Y, X)
        frame = frame[:, 0, :, :]
    else:
        frame = frame

    return frame
