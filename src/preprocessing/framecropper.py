import cv2
import os
import shutil

for label in [0, 1]:
    try:    
        shutil.rmtree(f"{label}_crop")
    except:
        pass
    os.mkdir(f"{label}_crop")


# SOURCE: 
# https://www.digitalocean.com/community/tutorials/how-to-detect-and-extract-faces-from-an-image-with-opencv-and-python


for label in [0, 1]:
    folder = os.listdir(f"./{label}")
    for img_name in folder:
        imagePath = f"./{label}/" + img_name

        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )

        print("[INFO] Found {0} Faces.".format(len(faces)))

        for (x, y, w, h) in faces:
            roi_color = image[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            cv2.imwrite(f'./{label}_crop/{w}{h}_{img_name}', roi_color)
            print(f"Saved cropped face at ./{label}_crop/{w}{h}_{img_name}")
