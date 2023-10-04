import numpy as np
import cv2


def print_table_row(name, np_array):
    print(f'{name:^{s[0]}}' f'{str(np_array.shape):^{s[1]}}' f'{np_array.ndim:^{s[2]}}')


# Generated ndarrays
m, n = 200, 600  # rows (height, y), columns(width, x) of generated numpy ndarrays
image2D_gray = np.random.randint(0, 255, (m, n), dtype=np.uint8)
image3D_gray = np.random.randint(0, 255, (m, n, 1), dtype=np.uint8)
image3D_color = np.random.randint(0, 255, (m, n, 3), dtype=np.uint8)
video4D_color = np.random.randint(0, 255, (m, n, 3, 300), dtype=np.uint8)
video3D_gray = np.random.randint(0, 255, (m, n, 300), dtype=np.uint8)

# Imported Image
cv2_logo = cv2.imread("OpenCV_Logo.png")
cv2_logo_greyscale = cv2.cvtColor(cv2_logo, cv2.COLOR_BGR2GRAY)

# This code is for png files (RGBA)
"""
cv2_logo_RGBA = cv2.imread("OpenCV_Logo_transparent_background.png", flags=cv2.IMREAD_UNCHANGED)
# Preserving Alpha channel during Grayscale require extra step
# Based on this forum post: https://answers.opencv.org/question/226779/create-2-channel-image-grayscale-and-alpha/
alpha = cv2_logo_RGBA[:, :, 3]
cv2_logo_RGBA_greyscale = cv2.cvtColor(cv2_logo_RGBA, cv2.COLOR_RGBA2GRAY)  # This will remove the alpha channel
cv2_logo_RGBA_greyscale = np.reshape(cv2_logo_RGBA_greyscale, (*cv2_logo_RGBA_greyscale.shape, 1))  # 2D -> 3D
alpha = np.reshape(alpha, (*alpha.shape, 1))
cv2_logo_RGBA_greyscale = np.concatenate((cv2_logo_RGBA_greyscale, alpha), axis=2)
# Note: 2 Channel Grey-Alpha is not supported by cv2.imshow
"""

s = (40, 28, 14)  # The table separation for each column
print(f'{"Array Name":^{s[0]}}' f'{"Shape":^{s[1]}}' f'{"ndim":^{s[2]}}')
print("=" * sum(s))
print_table_row("2D Grayscale (Image, Scalar)", image2D_gray)
print_table_row("3D Grayscale (Image, Tensor)", image3D_gray)
print_table_row("3D Color (Image)", image3D_color)
print_table_row("4D Color (Video)", video4D_color)
print_table_row("3D Gray (Video, Scalar)", video3D_gray)
print_table_row("Imported (Color) Image", cv2_logo)
print_table_row("Greyscale Imported Image", cv2_logo_greyscale)
# print_table_row("Imported RGBA Image", cv2_logo_RGBA)
# print_table_row("Greyscale Imported RGBA Image", cv2_logo_RGBA_greyscale)
print("\n*Press \"q\" while an image/video window is selected to exit")

# Display Generated ndarrays
cv2.imshow('2D Grayscale (Image, Scalar)', image2D_gray)
cv2.imshow('3D Grayscale (Image, Tensor)', image3D_gray)
cv2.imshow('3D Color (Image)', image3D_color)
# Display Image Imports
cv2.imshow("Imported (Color) Image", cv2_logo)
cv2.imshow("Greyscale Imported Image", cv2_logo_greyscale)
"""
cv2.imshow("Imported RGBA Image", cv2_logo_RGBA)
# Note: 2 Channel Grey-Alpha is not supported by cv2.imshow
# cv2.imshow("Greyscale RGBA Imported Image", cv2_logo_RGBA_greyscale)
"""
# Display Video
i = 0
while True:
    frame_c = i % video4D_color.shape[3]
    frame_g = i % video3D_gray.shape[2]
    cv2.imshow('4D Color (Video)', video4D_color[..., frame_c])  # tensor[:, :, :, frame] == tensor[..., frame]
    cv2.imshow('3D Gray (Video, Scalar)', video3D_gray[..., frame_g])
    if cv2.waitKey(100) == ord('q'):  # wait 100ms (10 Hz)
        break
    i += 1

cv2.destroyAllWindows()
