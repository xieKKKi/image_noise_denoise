import cv2
import numpy as np
import os


def normalize(mask):
    return (mask - mask.min()) / (mask.max() - mask.min())


def add_gaussian_noise(img, mu=0, sigma=0.1):
    image = np.array(img, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    image_out = image + noise
    image_out = np.uint8(normalize(image_out) * 255)
    return image_out


def add_rayleigh_noise(img, a=15):
    image = np.array(img, dtype=float)
    noise = np.random.rayleigh(a, size=image.shape)
    image_out = image + noise
    image_out = np.uint8(normalize(image_out) * 255)
    return image_out


def add_gamma_noise(img, scale=1):
    image = np.array(img, dtype=float)
    noise = np.random.gamma(shape=1, scale=scale, size=image.shape)
    image_out = image + noise
    image_out = np.uint8(normalize(image_out) * 255)
    return image_out


def add_exponent_noise(img, scale=1.0):
    image = np.array(img, dtype=float)
    noise = np.random.exponential(scale=scale, size=image.shape)
    image_out = image + noise
    image_out = np.uint8(normalize(image_out) * 255)
    return image_out


def add_average_noise(img, mean=0, sigma=100):
    image = np.array(img, dtype=float)
    a = 2 * mean - np.sqrt(12 * sigma)
    b = 2 * mean + np.sqrt(12 * sigma)
    noise = np.random.uniform(a, b, image.shape)
    image_out = image + noise
    image_out = np.uint8(normalize(image_out) * 255)
    return image_out


def add_salt_pepper(img, ps=0.01, pp=0.01):
    h, w = img.shape[:2]
    mask = np.random.choice((0, 0.5, 1), size=(h, w), p=[pp, (1 - ps - pp), ps])
    img_out = img.copy()
    img_out[mask == 1] = 255
    img_out[mask == 0] = 0
    return img_out


imgPath = "./origin.png"
noiseImgPath = './result/noiseImg/'
denoiseImgPath = './result/denoiseImg/'
if __name__ == "__main__":
    img = cv2.imread(imgPath)
    if not os.path.exists(noiseImgPath):
        os.makedirs(noiseImgPath)
    if not os.path.exists(denoiseImgPath):
        os.makedirs(denoiseImgPath)

    gauss_noiseImg = add_gaussian_noise(img, 0, 10)
    salt_pepper_noiseImg = add_salt_pepper(img, 0.01, 0.01)
    rayleigh_noiseImg = add_rayleigh_noise(img, 15)
    gamma_noiseImg = add_gamma_noise(img, 10)
    exponent_noiseImg = add_exponent_noise(img, 10)
    average_noiseImg = add_average_noise(img, 0, 50)
    cv2.imwrite(noiseImgPath + 'gauss_noiseImg.jpg', gauss_noiseImg)
    cv2.imwrite(noiseImgPath + 'salt_pepper_noiseImg.jpg', salt_pepper_noiseImg)
    cv2.imwrite(noiseImgPath + 'rayleigh_noiseImg.jpg', rayleigh_noiseImg)
    cv2.imwrite(noiseImgPath + 'gamma_noiseImg.jpg', gamma_noiseImg)
    cv2.imwrite(noiseImgPath + 'exponent_noiseImg.jpg', exponent_noiseImg)
    cv2.imwrite(noiseImgPath + 'average_noiseImg.jpg', average_noiseImg)

    denoise_method = ['meanBlur', 'boxFilter', 'GaussianBlur', 'medianBlur', 'NonLocalMeans']
    for method in denoise_method:
        if method == 'meanBlur':
            gauss_denoiseImg = cv2.blur(gauss_noiseImg, (5, 5))
            salt_pepper_denoiseImg = cv2.blur(salt_pepper_noiseImg, (5, 5))
            rayleigh_denoiseImg = cv2.blur(rayleigh_noiseImg, (5, 5))
            gamma_denoiseImg = cv2.blur(gamma_noiseImg, (5, 5))
            exponent_denoiseImg = cv2.blur(exponent_noiseImg, (5, 5))
            average_denoiseImg = cv2.blur(average_noiseImg, (5, 5))
        elif method == 'boxFilter':
            gauss_denoiseImg = cv2.boxFilter(gauss_noiseImg, -1, (5, 5), normalize=1)
            salt_pepper_denoiseImg = cv2.boxFilter(salt_pepper_noiseImg, -1, (5, 5), normalize=1)
            rayleigh_denoiseImg = cv2.boxFilter(rayleigh_noiseImg, -1, (5, 5), normalize=1)
            gamma_denoiseImg = cv2.boxFilter(gamma_noiseImg, -1, (5, 5), normalize=1)
            exponent_denoiseImg = cv2.boxFilter(exponent_noiseImg, -1, (5, 5), normalize=1)
            average_denoiseImg = cv2.boxFilter(average_noiseImg, -1, (5, 5), normalize=1)
        elif method == 'GaussianBlur':
            gauss_denoiseImg = cv2.GaussianBlur(gauss_noiseImg, (5, 5), 0)
            salt_pepper_denoiseImg = cv2.GaussianBlur(salt_pepper_noiseImg, (5, 5), 0)
            rayleigh_denoiseImg = cv2.GaussianBlur(rayleigh_noiseImg, (5, 5), 0)
            gamma_denoiseImg = cv2.GaussianBlur(gamma_noiseImg, (5, 5), 0)
            exponent_denoiseImg = cv2.GaussianBlur(exponent_noiseImg, (5, 5), 0)
            average_denoiseImg = cv2.GaussianBlur(average_noiseImg, (5, 5), 0)
        elif method == 'medianBlur':
            gauss_denoiseImg = cv2.medianBlur(gauss_noiseImg, 5)
            salt_pepper_denoiseImg = cv2.medianBlur(salt_pepper_noiseImg, 5)
            rayleigh_denoiseImg = cv2.medianBlur(rayleigh_noiseImg, 5)
            gamma_denoiseImg = cv2.medianBlur(gamma_noiseImg, 5)
            exponent_denoiseImg = cv2.medianBlur(exponent_noiseImg, 5)
            average_denoiseImg = cv2.medianBlur(average_noiseImg, 5)
        # elif method == 'Non-Local Means':
        else:
            gauss_denoiseImg = cv2.fastNlMeansDenoisingColored(gauss_noiseImg)
            salt_pepper_denoiseImg = cv2.fastNlMeansDenoisingColored(salt_pepper_noiseImg)
            rayleigh_denoiseImg = cv2.fastNlMeansDenoisingColored(rayleigh_noiseImg)
            gamma_denoiseImg = cv2.fastNlMeansDenoisingColored(gamma_noiseImg)
            exponent_denoiseImg = cv2.fastNlMeansDenoisingColored(exponent_noiseImg)
            average_denoiseImg = cv2.fastNlMeansDenoisingColored(average_noiseImg)
        cv2.imwrite(denoiseImgPath + method + '_' + 'gauss_denoiseImg.jpg', gauss_denoiseImg)
        cv2.imwrite(denoiseImgPath + method + '_' + 'salt_pepper_denoiseImg.jpg', salt_pepper_denoiseImg)
        cv2.imwrite(denoiseImgPath + method + '_' + 'rayleigh_denoiseImg.jpg', rayleigh_denoiseImg)
        cv2.imwrite(denoiseImgPath + method + '_' + 'gamma_denoiseImg.jpg', gamma_denoiseImg)
        cv2.imwrite(denoiseImgPath + method + '_' + 'exponent_denoiseImg.jpg', exponent_denoiseImg)
        cv2.imwrite(denoiseImgPath + method + '_' + 'average_denoiseImg.jpg', average_denoiseImg)
