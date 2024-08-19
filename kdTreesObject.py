import numpy as np
import cv2
import matplotlib.pyplot as plt
from webcolors import rgb_to_hex
import heapq as hq 

#https://www.baeldung.com/cs/k-d-trees#:~:text=The%20process%20starts%20by%20selecting,relative%20to%20the%20splitting%20hyperplane.
#https://www.ri.cmu.edu/pub_files/pub1/moore_andrew_1991_1/moore_andrew_1991_1.pdf
class KdNode(object):
	"""docstring for KdTree"""
	def __init__(self, value, axis, image_view):
		super(KdNode, self).__init__()
		self.value = value
		self.axis = axis
		self._left = None
		self._right = None
		self.image_view = image_view


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
	def __init__(self, image, colors):
		super(KdTree, self).__init__()
		self.image_org = image.copy()
		self.image = image.reshape(-1, 3)

		self.colors = colors
		self.median_list = []
		self.axis_list = []
		self.slice_queue = self.get_slices(self.image_org)
		self.kd_tree = self.create_kd_tree(self.image, self.colors, self.image_org)

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
		covariances_only = []
		for i in range(3):
			#print(image.shape, np.min(image[:,i]))
			min_range = np.min(image[:,i]) + 1
			max_range = np.max(image[:,i])
			otsu_vars = []
			otsu_vars_only = []

			for th in range(min_range, max_range):
				var = otsu_intraclass_variance(image[:,i], th)
				otsu_vars.append((var, (i, th)))
				otsu_vars_only.append(var)
			covariances = covariances + (otsu_vars)
			covariances_only += otsu_vars_only
		
		
		ret_cov = None
		if len(covariances) > 0:
			cov_array = np.asarray(covariances_only)
			ret_cov = covariances[cov_array.argmin()]
			#print(ret_cov)

		#fig, ax_lst = plt.subplots(1, 3)
		#ax_lst[0].scatter(range(len(covariances_only[0])), covariances_only[0])
		#ax_lst[1].scatter(range(len(covariances_only[1])), covariances_only[1])
		#ax_lst[2].scatter(range(len(covariances_only[2])), covariances_only[2])
		#plt.show()

		
		return ret_cov


	def create_kd_tree(self, image, colors, crop_image=None):
		headNode = None

		if colors <= 0:
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


		slice_heap = []

		image_cur = image

		slice_queue_instance = self.get_slices(image_cur)

		axis = slice_queue_instance[1][0]
		image = image[image[:, axis].argsort()]
		threshold = slice_queue_instance[1][1]

		headNode = KdNode(threshold, axis, image_cur)
		cur_node = headNode

		heap_value = (slice_queue_instance[0], (slice_queue_instance[1], headNode))
		hq.heappush(slice_heap, heap_value)

		for i in range(colors):

			cur_node = hq.heappop(slice_heap)
			print("CURNODE", cur_node)
			(variance, ((axis, threshold) , cur_kd_node)) = cur_node

			

			if slice_queue_instance is None:
				return None

		
			image = cur_kd_node.image_view[image[:, axis].argsort()]
			
			self.axis_list.append(axis)
			self.median_list.append((threshold, colors))

			left_array_arg = image[image[:, axis] < threshold]
			right_array_arg = image[image[:, axis] >= threshold]

			if len(left_array_arg) > 0:
				slice_queue_instance_left = self.get_slices(left_array_arg)

				axis = slice_queue_instance_left[1][0]
				#image = left_array_arg[left_array_arg[:, axis].argsort()]
				threshold = slice_queue_instance_left[1][1]
				leftNode = KdNode(threshold, axis, left_array_arg)
				cur_kd_node.set_left(leftNode)

				heap_value = (slice_queue_instance_left[0], (slice_queue_instance_left[1], leftNode))
				print(heap_value)
				hq.heappush(slice_heap, heap_value)

			if len(right_array_arg) > 0:
				slice_queue_instance_right = self.get_slices(right_array_arg)

				axis = slice_queue_instance_right[1][0]
				image = left_array_arg[left_array_arg[:, axis].argsort()]
				threshold = slice_queue_instance_right[1][1]
				rightNode = KdNode(threshold, axis, right_array_arg)
				cur_kd_node.set_right(rightNode)

				heap_value = (slice_queue_instance_right[0], (slice_queue_instance_right[1], rightNode))
				hq.heappush(slice_heap, heap_value)





		return headNode


	def find_mapping_value(self, value):
		depth_limit = self.colors
		cur_node = self.kd_tree
		#value = None
		dimension = 0
		ret = None
		pixel_values_ret = [value[0], value[1], value[2]]

		while not cur_node is None:
			cur_val = cur_node.value
			axis = cur_node.axis
			left = cur_node.get_left()
			right = cur_node.get_right()
			ret = cur_node
			pixel_values_ret[axis] = cur_val

			if value[axis] < cur_val:
				#print(value[axis], cur_val, "left", str(axis))
				cur_node = left
			else:
				#print(value[axis], cur_val, "right", str(axis))
				cur_node = right
			dimension += 1

		return pixel_values_ret

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
	print(m_list)
	#print(m_list - np.array([0, 125, 239]))
	#print(kd_tree.find_mapping_value([0, 125, 239]))
	quantize_output = np.zeros(quantize_input.shape)
	print(quantize_input[350,250])
	print(kd_tree.find_mapping_value(quantize_input[350,250]))

	for i in range(quantize_input.shape[0]):
		for j in range(quantize_input.shape[1]):
			quantize_output[i, j] = kd_tree.find_mapping_value(quantize_input[i,j])
	
	quantize_output = quantize_output.astype('uint8') 
	#quantize_output = cv2.cvtColor(quantize_output, cv2.COLOR_BGR2RGB)
	print(quantize_input[350,250])
	print("Quantized")
	print(quantize_output[350,250])
	#plt.imshow(quantize_output)
	#plt.show()
	cv2.imshow("quant", quantize_output)
	cv2.waitKey(0)

	plt.show()
if __name__ == '__main__':

	main()