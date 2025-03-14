import opendatasets as od
import test_filter as tf
import test_filter_sebas as tfs

tf.findMeningiomaByContours('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0015.jpg')

tf.findMeningiomaByContours('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0010.jpg')

tfs.bg_image('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0010.jpg')

tf.find_meningioma_by_contour_size('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0015.jpg', 10)

tf.find_meningioma_rect_crop('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0246.jpg',2.779)


#tf.region_growing('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0015.jpg', seedpoint, threshold=0.1)