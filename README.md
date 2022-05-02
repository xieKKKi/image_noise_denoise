噪声类型：高斯噪音、瑞利噪音、爱尔兰（伽马）噪音、指数噪音、均匀噪音、椒盐噪音

去噪手段：'meanBlur', 'boxFilter', 'GaussianBlur', 'medianBlur', 'NonLocalMeans'

原图：origin.png

源码：denoise.py

结果：result文件夹：

* noiseImg中是添加噪声的图片，命名格式为：**噪声类型-noiseImg.jpg**
* denoiseImg中是使用各种方法去噪声的图片，命名格式为：**去噪方法-噪声类型-denoiseImg.jpg**

