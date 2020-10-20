import numpy as np
from PIL import Image


def pad_img_KAR(img, target_w, target_h, pad_value=(124, 116, 104)):
	w, h = img.size
	if w / h == target_w / target_h:
		return img, (0, 0)

	if w < h or w * (target_h / target_w) < h:
		new_w = int(h * (target_w / target_h))
		new_img = Image.new('RGB', (new_w, h), color=pad_value)
		new_img.paste(img, (int((new_w - w) // 2), 0))
		return new_img, (new_w - w, 0)
	else:
		new_h = int(w * (target_h / target_w))
		new_img = Image.new('RGB', (w, new_h), color=pad_value)
		new_img.paste(img, (0, int((new_h - h) // 2)))
		return new_img, (0, new_h - h)


def pad_array_KAR(arr, target_h, target_w, pad_value=np.array([[0]])):
	h, w = arr.shape
	if w / h == target_w / target_h:
		return arr, (0, 0)

	if w < h or w * (target_h / target_w) < h:
		new_w = int(h * (target_w / target_h))
		new_arr = np.ones((h, new_w)) * pad_value
		new_arr[:, int((new_w - w) // 2):int((new_w - w) // 2)+w] = arr
		return new_arr, (0, new_w - w)
	else:
		new_h = int(w * (target_h / target_w))
		new_arr = np.ones((new_h, w)) * pad_value
		new_arr[int((new_h - h) // 2):int((new_h - h) // 2) + h, :] = arr
		return new_arr, (new_h - h, 0)
