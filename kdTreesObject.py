import numpy as np
import cv2
import matplotlib.pyplot as plt
from webcolors import rgb_to_hex
import heapq as hq 

#https://www.baeldung.com/cs/k-d-trees#:~:text=The%20process%20starts%20by%20selecting,relative%20to%20the%20splitting%20hyperplane.
#https://www.ri.cmu.edu/pub_files/pub1/moore_andrew_1991_1/moore_andrew_1991_1.pdf
class KdNode(object):
	"""docstring for KdTree"""
	def __init__(self, value, axis):
		super(KdNode, self).__init__()
		self.value = value
		self.axis = axis
		self._left = None
		self._right = None


	def get_left(self):
		return self._left

	def get_right(self):
		return self._right

	def set_left(self, left):
		self._left = left

	def set_right(self, right):
		self._right = right

		


class KdTree(object):
	"""docstring for KdTree"""
	def __init__(self, image, depth_limit):
		super(KdTree, self).__init__()
		self.image_org = image.copy()
		self.image = image.reshape(-1, 3)

		self.depth_limit = depth_limit
		self.median_list = []
		self.axis_list = []
		#self.slice_queue = self.get_slices(self.image_org)
		self.kd_tree = self.create_kd_tree(self.image, self.depth_limit, self.image_org)

	def get_slices(self, image=None):


		def otsu_intraclass_variance(image, threshold):
		    """
		    Otsu's intra-class variance.
		    If all pixels are above or below the threshold, this will throw a warning that can safely be ignored.
		    """
		    return np.nansum(
		        [
		            np.mean(cls) * np.var(image, where=cls)
		            #   weight   ·  intra-class variance
		            for cls in [image >= threshold, image < threshold]
		        ]
		    ) 

		covariances = []
		
		for i in range(3):
			print(image.shape, np.min(image[:,:,i]))
			min_range = np.min(image[:,:,i]) + 1
			max_range = np.max(image[:,:,i])
			otsu_vars = []
			for th in range(min_range, max_range):
				var = otsu_intraclass_variance(image[:,:,i], th)
				otsu_vars.append((var, i))
			covariances = covariances + (otsu_vars)

		hq.heapify(covariances) 
		print(covariances[0])
		return covariances


	def create_kd_tree(self, image, depth_limit, crop_image=None):
		headNode = None

		if depth_limit <= 0:
			return None

		if image is None:
			# Would rather have an error message
			return None

		if len(image) == 0:
			return None

		
		left_array_arg = None
		right_array_arg = None
		cov = None
		axis_final = None

		if crop_image is None:
			crop_image = self.image_org.copy()

		axis = self.get_slices(crop_image)[0][1]

		sortted_array = image[image[:, axis].argsort()]
		

		median_index = len(sortted_array)//2
		median = image[median_index]
		median_index_final = image[:, axis].tolist().index(median[axis])


		left_array_arg = image[:median_index_final]
		right_array_arg = image[median_index_final:]

		median_final = median
		axis_final = axis

		if axis == 0:
			crop_image_l = crop_image[:median_index_final,:,:]
			crop_image_r = crop_image[median_index_final:,:,:]
		elif axis == 1:
			crop_image_l = crop_image[:,:median_index_final,:]
			crop_image_r = crop_image[:,median_index_final:,:]
		else:
			crop_image_l = crop_image[:,:,:median_index_final]
			crop_image_r = crop_image[:,:,median_index_final:]

		self.axis_list.append(axis_final)
		self.median_list.append((median_final, depth_limit))
		cur_node = KdNode(median_final, axis_final)

		#print(median)

		cur_node.set_left(self.create_kd_tree(left_array_arg, depth_limit-1,crop_image_l))
		cur_node.set_right(self.create_kd_tree(right_array_arg, depth_limit-1, crop_image_r))

		return cur_node


	def find_mapping_value(self, value):
		depth_limit = self.depth_limit
		cur_node = self.kd_tree
		#value = None
		dimension = 0
		ret = None
		while not cur_node is None:
			cur_val = cur_node.value
			axis = cur_node.axis
			left = cur_node.get_left()
			right = cur_node.get_right()
			ret = cur_node

			if value[axis] <= cur_val[axis]:
				#print(cur_val, "left", str(axis))
				cur_node = left
			else:
				#print(cur_val, "right", str(axis))
				cur_node = right
			dimension += 1

		return ret.value

	def visualize_kd_slices(self, hsv=False):
		image = self.image.reshape(-1, 3)
		colors = np.apply_along_axis(rgb_to_hex, -1, image)
		dim1 = image[:, 0]
		dim2 = image[:, 1]
		dim3 = image[:, 2]
		ax = plt.gca(projection="3d")
		#ax.scatter(dim1, dim2, dim3, c=colors)
		for idx, (median, depth) in enumerate(self.median_list):
			index = self.axis_list[idx]
			print(median, index)
			depth = depth - self.depth_limit
			# if idx != 0:
			#axes_dim = np.linspace(median[index] - 256 * 2**(depth -1), median[index] +  256 * 2**(depth -1), 255)
			static_dim1 = np.linspace(median[(index + 1) % 3], median[(index + 1) % 3], 255)
			static_dim2 = np.linspace(median[(index + 2) % 3], median[(index + 2) % 3], 255)
			# else:

			axes_dim = np.linspace(0, 255, 255)
			# static_dim1 = np.linspace(median[(index + 1) % 3], median[(index + 1) % 3], 255)
			# static_dim2 = np.linspace(median[(index + 2) % 3], median[(index + 2) % 3], 255)

			lines_values = [None, None, None]

			lines_values[index] = axes_dim
			lines_values[(index + 1) % 3] = static_dim1
			lines_values[(index + 2) % 3] = static_dim2

			#print(lines_values)
			ax.plot(lines_values[0], lines_values[1], lines_values[2])


		
		plt.show()

def main():
	quantize_input = cv2.imread("quantize_input.png")
	kd_tree = KdTree(quantize_input, 5)
	#kd_tree.visualize_kd_tree()
	m_list = [m for m, d in kd_tree.median_list]
	m_list = np.vstack(m_list)
	#print(m_list)
	#print(m_list - np.array([0, 125, 239]))
	#print(kd_tree.find_mapping_value([0, 125, 239]))
	quantize_output = np.zeros(quantize_input.shape)
	print(quantize_input[0,0])
	kd_tree.find_mapping_value(quantize_input[0,0])

	for i in range(quantize_input.shape[0]):
		for j in range(quantize_input.shape[1]):
			quantize_output[i, j] = kd_tree.find_mapping_value(quantize_input[i,j])
	
	quantize_output = quantize_output.astype('uint8') 
	#quantize_output = cv2.cvtColor(quantize_output.astype('uint8'), cv2.COLOR_BGR2RGB)
	print(quantize_input[0,0])
	print("Quantized")
	print(quantize_output[0,0])
	cv2.imshow("quant", quantize_output)
	cv2.waitKey(0)

	plt.show()
if __name__ == '__main__':

	main()