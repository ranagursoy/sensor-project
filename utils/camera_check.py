import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Kamera {i} açık!")
        cap.release()
    else:
        print(f"Kamera {i} mevcut değil.")

image = cv2.imread("frames\camera0_0.png")
height, width, channels = image.shape
print(f"Görüntü boyutları: Genişlik = {width}, Yükseklik = {height}, Kanallar = {channels}")
