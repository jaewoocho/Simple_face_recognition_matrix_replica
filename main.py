# 1. Import Libraries and import models 
# dlib: face detection + face recognition 
import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# Face landmark prediction model(used in face overlap project)
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('Desktop/model/shape_predictor_68_face_landmarks.dat')

#Face recognition model
facerec = dlib.face_recognition_model_v1('Desktop/model/dlib_face_recognition_resnet_model_v1.dat')

_______________________________________________________________________________________________________________________________________________________________________
# 2. Face Dectection Function and Face Encoding Function
# Face detection function

#Function that finds the face as a image 
def find_faces(img):
    dets = detector(img, 1)

    ## When face detection failes, empty arrays are retruend and logic ends
    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    # As the model detects a face, it wraps it with a rectangle
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)
        
        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
        
    return rects, shapes, shapes_np


# Face encoding funciton
# Step 1: The model will detect facial features as landmark points
# Step 2: These landmarks are placed in a encoder that will be converted into 128 vectors
# Step 3: These 128 vecotrs will be used to distringuish and recognize a specific face (ex. Neo)

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)
    
    
_______________________________________________________________________________________________________________________________________________________________________
# 3. Compute Saved Face Descriptions

#Recalling Matrix images from Path (Jupyter notebook)
 img_paths = {
    'neo': 'Desktop/model/neo.jpg',
    'trinity': 'Desktop/model/trinity.jpg',
    'morpheus': 'Desktop/model/morpheus.jpg',
    'smith': 'Desktop/model/smith.jpg'
}

descs = {
    'neo': None,
    'trinity': None,
    'morpheus': None,
    'smith': None
}

# Read img from path
for name, img_path in img_paths.items():
    
    # Changing bgr to rbg format
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Input the image and landmarks of each image inside the eoncoding function
    _, img_shapes, _ = find_faces(img_rgb)
    descs[name] = encode_faces(img_rgb, img_shapes)[0]

# Save the results as  in a numpy array
#The iamge is shown in numy array values
np.save('Desktop/model/descs.npy', descs)
print(descs)


_______________________________________________________________________________________________________________________________________________________________________
# 4. Compute input for testing 
# Recall a different image and change the values from bgr to rbg
img_bgr = cv2.imread('img/matrix5.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)


_______________________________________________________________________________________________________________________________________________________________________
# 5. Final Testing
#Visualize Output
fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

#Computes loops per description
for i, desc in enumerate(descriptors):
    
    found = False
    for name, saved_desc in descs.items():
        #Finds the different in face landmarks in euclidean distance for face recogntion
        dist = np.linalg.norm([desc] - saved_desc, axis=1)
        
        #Euclidean distance from face landmark (K-nearest neigbor)
        # 0.6 is the ideal distance for this model
        if dist < 0.6:
            found = True

            # If face is detected write the name of the face and draw a rectangle 
            text = ax.text(rects[i][0][0], rects[i][0][1], name,
                    color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)

            break
      #If face isn't detected, identify the name as "unknown" and draw rectangle 
    if not found:
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                             rects[i][1][1] - rects[i][0][1],
                             rects[i][1][0] - rects[i][0][0],
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
plt.savefig('result/output.png')
plt.show()
