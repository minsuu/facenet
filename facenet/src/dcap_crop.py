"""Performs face alignment and stores face thumbnails in the output directory."""

from scipy import misc
import os
import sys
import align_dlib
import random
import json

PREFIX = os.getcwd()

input_dir = PREFIX + '/vid' # Directory with unaligned images
dlib_face_predictor = PREFIX + '/data/models/shape_predictor_68_face_landmarks.dat'
                           # File containing the dlib face predictor
image_size = 110 # Image size (height, width) in pixels
face_size = 96 # Size of the face thumbnail (height, width) in pixels
use_new_alignment = False # Indicates if the improved alignment transformation should be used.""")

def crop(name):
    align = align_dlib.AlignDlib(os.path.expanduser(dlib_face_predictor))
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE
    base_dir = os.path.join(os.path.expanduser(input_dir),name)
    scale = float(face_size) / image_size
    if os.path.exists(base_dir):
        bb_filename = os.path.join(base_dir, 'bb.json')
#        if os.path.exists(bb_filename):
#            continue
        bb_list = []
        images = os.listdir(base_dir)
        for image_path in sorted(map(lambda x: os.path.join(base_dir, x), images)):
            if(os.path.splitext(image_path)[1] == ".png"):
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                # find the bounding box...
                print("finding face in " + image_path)
                img = misc.imread(image_path)
                bb = align.getLargestFaceBoundingBox(img, False)
                if bb is None:
                    continue
                print("found!")
                bb = [ bb.left(), bb.top(), bb.width(), bb.height() ]
                bb_list.append( { 'name' : filename+'.png', 'bb' : bb } )
                continue
        with open(bb_filename, 'w') as f:
            json.dump(bb_list,f)

if __name__ == '__main__':
    crop(sys.argv[1])
