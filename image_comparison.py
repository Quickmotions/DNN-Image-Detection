# image_comparison.py -- Fergus Haak -- 14/09/2023

from PIL import Image
from PIL import ImageChops
import imagehash


def image_pixel_similarity(base_image, compare_image):
  hash0 = imagehash.average_hash(base_image)
  hash1 = imagehash.average_hash(compare_image)
  # cutoff = 5  # maximum bits that could be different between the hashes.
  return hash0-hash1



def image_pixel_differences(base_image, compare_image):
  diff = ImageChops.difference(base_image, compare_image)
  diff.show()

base_image = Image.open('images/iona1.jpg')
compare_image = Image.open('images/iona2.jpg')
image_pixel_differences(base_image, compare_image)
results = image_pixel_similarity(base_image, compare_image)
print(results)