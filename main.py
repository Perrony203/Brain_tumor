import opendatasets as od
import test_filter as tf
import test_filter_sebas as tfs
'''
tf.highlight_meningiomas('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0010.jpg')

tf.display_equalized_image('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0010.jpg')

tf.display_sobel_edges('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0010.jpg')
'''

#seedpoint = tf.preprocess_and_find_white_pixel('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0015.jpg')

#tf.sobelpluswhite('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0012.jpg')
tfs.bg_image('brain-tumor-mri-dataset/Testing/meningioma/Te-meTr_0003.jpg')
#tf.region_growing('brain-tumor-mri-dataset/Testing/meningioma/Te-me_0015.jpg', seedpoint, threshold=0.1)