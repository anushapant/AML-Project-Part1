import cv2
import matplotlib.pyplot as plt

FD = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def blurFaces(img, fname):
    faces = FD.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        face = img[y:y + h, x:x + w]
        height, width = face.shape[0], face.shape[1]
        img[y:y + height, x:x + width] = cv2.GaussianBlur(face, (95, 95), 30)

    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.show()
    filename = fname + "Blurred.png"
    print(filename)
    cv2.imwrite(str(filename), img)
    print("done")


file = 'Humans/1 (1).jpeg'
image = cv2.cvtColor(cv2.imread(file), cv2.IMREAD_GRAYSCALE)
name = file.split("/")[1]
name = name.split(".")[0]
blurFaces(image, name)