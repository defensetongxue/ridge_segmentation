from PIL import Image
import numpy as np

m1 = Image.open('./mask.png')
m2 = Image.open('./processed_mask.png')
m1 = np.array(m1)
m2 = np.array(m2)
m1[m1 != 0] = 1
m2[m2 != 0] = 1
m1 = m1 - m2
res = np.where(m1 != 0, 1, 0)

# Convert 'res' to uint8 data type
res_uint8 = res.astype(np.uint8) * 255

# Save the result
Image.fromarray(res_uint8).save('./res.png')
