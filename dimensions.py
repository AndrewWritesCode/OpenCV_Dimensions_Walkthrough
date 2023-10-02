import numpy as np
import cv2


image2D_gray = np.random.randint(0, 255, (200, 700), dtype=np.uint8)
image3D_gray = np.random.randint(0, 255, (200, 700, 1), dtype=np.uint8)
image3D_color = np.random.randint(0, 255, (200, 700, 3), dtype=np.uint8)
video4D_color = np.random.randint(0, 255, (200, 700, 3, 300), dtype=np.uint8)
video3D_gray = np.random.randint(0, 255, (200, 700, 300), dtype=np.uint8)

print(f'{"Array Name":^40}' f'{"Shape":^30}' f'{"ndim":^20}')
print("="*90)
print(f'{"2D Grayscale (Image, Scalar)":^40}' f'{str(image2D_gray.shape):^30}' f'{image2D_gray.ndim:^20}')
print(f'{"3D Grayscale (Image, Tensor)":^40}' f'{str(image3D_gray.shape):^30}' f'{image3D_gray.ndim:^20}')
print(f'{"3D Color (Image)":^40}' f'{str(image3D_color.shape):^30}' f'{image3D_color.ndim:^20}')
print(f'{"4D Color (Video)":^40}' f'{str(video4D_color.shape):^30}' f'{video4D_color.ndim:^20}')
print(f'{"3D Gray (Video, Scalar)":^40}' f'{str(video3D_gray.shape):^30}' f'{video3D_gray.ndim:^20}')
print("\n Press \"q\" while images/video windows are selected to exit")

cv2.imshow('2D Grayscale (Image, Scalar)', image2D_gray)
cv2.imshow('3D Grayscale (Image, Tensor)', image3D_gray)
cv2.imshow('3D Color (Image)', image3D_color)
# Display Video
i = 0
while True:
    frame_c = i % video4D_color.shape[3]
    frame_g = i % video3D_gray.shape[2]
    cv2.imshow('4D Color (Video)', video4D_color[:, :, :, frame_c])
    cv2.imshow('3D Gray (Video, Scalar)', video3D_gray[:, :, frame_g])
    if cv2.waitKey(100) == ord('q'):  # wait 100ms (10 Hz)
        break
    i += 1

cv2.destroyAllWindows()
