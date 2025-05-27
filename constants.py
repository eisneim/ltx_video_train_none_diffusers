
import os


DEFAULT_HEIGHT_BUCKETS = [ x * 32 for x in list(range(7, 50))]
DEFAULT_WIDTH_BUCKETS = [ x * 32 for x in list(range(7, 50))]
DEFAULT_FRAME_BUCKETS = [ ii * 8 + 1 for ii in range(3, 30) ]

DEFAULT_IMAGE_RESOLUTION_BUCKETS = []
for height in DEFAULT_HEIGHT_BUCKETS:
    for width in DEFAULT_WIDTH_BUCKETS:
        DEFAULT_IMAGE_RESOLUTION_BUCKETS.append((height, width))

DEFAULT_VIDEO_RESOLUTION_BUCKETS = []
for frames in DEFAULT_FRAME_BUCKETS:
    for height in DEFAULT_HEIGHT_BUCKETS:
        for width in DEFAULT_WIDTH_BUCKETS:
            DEFAULT_VIDEO_RESOLUTION_BUCKETS.append((frames, height, width))
