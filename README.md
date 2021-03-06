# kmeans
Extremely simple implementation of KMeans clustering algorithm

## Usage

First, you create virtual environment and install package.
I already tested it and it works well in my environment (Windows 10).
```
conda create python=3.8 -n kmeans
conda activate kmeans
pip install numpy matplotlib
python kmeans.py
```

For your information, I attached a help message for better usage.
```
usage: kmeans.py [-h] [--k K] [--input INPUT]

optional arguments:
  -h, --help     show this help message and exit
  --k K          The number of centroid (default to 50)
  --input INPUT  The filepath for input image (default to os.path.join("res", "input.jpg"))
```

## Input

I just downloaded an image from the image search result in Google with keyword "Detroit become human kara".
This image could raise an copyright issue.
If it occurs, I will change the other copyright-free image, but for now, I want to keep this image because this image makes me passionate to implement this code.

The original source: https://www.syfy.com/syfywire/video-game-heroine-of-the-month-kara-from-detroit-become-human
![input](https://user-images.githubusercontent.com/12251974/109493461-ded12700-7acf-11eb-8c73-782f20bcfc7a.jpg)

## Output

| K | Result          | K | Result |
|:-:|:---------------:|:-:|:------:|
| 2 | ![result_2](https://user-images.githubusercontent.com/12251974/109493389-c3feb280-7acf-11eb-8f67-bc30dd332350.gif) | 3 | ![result_3](https://user-images.githubusercontent.com/12251974/109493400-c82ad000-7acf-11eb-8603-4f300f0c3a74.gif) |
| 4 | ![result_4](https://user-images.githubusercontent.com/12251974/109493403-c9f49380-7acf-11eb-8714-b86ee5704567.gif) | 5 | ![result_5](https://user-images.githubusercontent.com/12251974/109493406-cbbe5700-7acf-11eb-97fd-61ef2f8a4b3b.gif) |
| 6 | ![result_6](https://user-images.githubusercontent.com/12251974/109493409-ccef8400-7acf-11eb-8f5b-885f84dfc042.gif) | 7 | ![result_7](https://user-images.githubusercontent.com/12251974/109493410-ce20b100-7acf-11eb-94bf-c3cfa1ed5444.gif) |
| 8 | ![result_8](https://user-images.githubusercontent.com/12251974/109493411-cf51de00-7acf-11eb-92be-d433f89b2153.gif) | 9 | ![result_9](https://user-images.githubusercontent.com/12251974/109493417-d0830b00-7acf-11eb-94bc-dccb2dc67ccf.gif) |
| 10 | ![result_10](https://user-images.githubusercontent.com/12251974/109493420-d1b43800-7acf-11eb-86b1-a4ed9fe30152.gif) | 50 | ![result_50](https://user-images.githubusercontent.com/12251974/109493421-d2e56500-7acf-11eb-9bba-a2f6fc39a213.gif) |

## Contact

Please open an issue or send an e-mail if you have trouble with this code or find a bug.

* seonghyeon.drew@gmail.com
* sh0416@postech.ac.kr
