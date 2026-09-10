import albumentations as A
import numpy as np

img = np.zeros((10, 10, 3), dtype=np.uint8)
m1 = np.ones((10, 10), dtype=np.uint8)
m2 = np.ones((10, 10), dtype=np.uint8)

t = A.Compose([A.HorizontalFlip(p=1.0)])
res = t(image=img, masks=[m1, m2])
print("Success:", len(res['masks']))
