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
	def __init__(self, image, slices):
		super(KdTree, self).__init__()
		self.image_org = image.copy()
		self.image = image.reshape(-1, 3)

		self.slices = slices
		self.median_list = []
		self.axis_list = []
		#self.slice_queue = self.get_slices(self.image)
		self.slice_heap = None
		self.kd_tree = self.create_kd_tree(self.image, self.slices, self.image_org)
		

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
		ret_cov = None
		var_split_best_dif = None

		for i in range(3):
			#print(image.shape, np.min(image[:,i]))
			min_range = np.min(image[:,i]) + 1
			max_range = np.max(image[:,i])
			otsu_vars = []
			otsu_vars_only = []
			#print(min_range, max_range)
			for th in range(min_range, max_range):

				var = otsu_intraclass_variance(image[:,i], th)
				otsu_vars.append((var, (i, th)))
				otsu_vars_only.append(var)
			
			covariances = covariances + otsu_vars
			covariances_only.append(otsu_vars_only)

			if len(otsu_vars_only) > 0:
				cov_array = np.asarray(otsu_vars_only)
				cov_arg_index = np.argmin(cov_array)
				index = -1
				#temp_cov = otsu_vars[cov_arg_index][0]
				temp_cov =  np.mean(image[:,i]) * np.var(image[:,i])
				thresh = otsu_vars[cov_arg_index][1][1]
				
				temp_image_left =  image[image[:,i] < thresh][:,i]
				temp_image_right =  image[image[:,i] >= thresh][:,i]
				var_split = np.mean(temp_image_left) * np.var(temp_image_left) +   np.mean(temp_image_right) * np.var(temp_image_right)
				
				if ret_cov is None:
					#print(np.array(covariances)[cov_argsort[:5]])
					ret_cov = otsu_vars[cov_arg_index]
					var_split_best_dif = temp_cov - var_split
				else:
					if var_split_best_dif < (temp_cov - var_split):
						ret_cov = otsu_vars[cov_arg_index]
						var_split_best_dif = temp_cov - var_split
					elif var_split_best_dif == (temp_cov - var_split):
						if len(covariances_only[i]) >= len(covariances_only[i-1]):
							ret_cov = otsu_vars[cov_arg_index]
							var_split_best_dif = temp_cov - var_split
		
			#print(ret_cov)

		# fig, ax_lst = plt.subplots(2, 3)
		# bins = np.arange(0, 255, 1) # fixed bin size
		# ax_lst[0, 0].axis(xmin=min(image[:, 0])-5, xmax=max(image[:, 0])+5)
		# ax_lst[0, 1].axis(xmin=min(image[:, 1])-5, xmax=max(image[:, 1])+5)
		# ax_lst[0, 2].axis(xmin=min(image[:, 2])-5, xmax=max(image[:, 2])+5)
		# ax_lst[0, 0].hist(image[:, 0], bins=bins, alpha=0.5)
		# ax_lst[0, 1].hist(image[:, 1], bins=bins, alpha=0.5)
		# ax_lst[0, 2].hist(image[:, 2], bins=bins, alpha=0.5)

		# ax_lst[1, 0].scatter(range(len(covariances_only[0])), covariances_only[0])
		# ax_lst[1, 1].scatter(range(len(covariances_only[1])), covariances_only[1])
		# ax_lst[1, 2].scatter(range(len(covariances_only[2])), covariances_only[2])
		#print(ret_cov, var_split_best_dif)
		#plt.show()

		
		return ret_cov


	def create_kd_tree(self, image, slices, crop_image=None):
		headNode = None

		if slices <= 0:
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
		self.slice_heap = slice_heap
		image_cur = self.image

		slice_queue_instance = self.get_slices(image_cur)

		axis = slice_queue_instance[1][0]
		image = image[image[:, axis].argsort()]
		threshold = slice_queue_instance[1][1]
		difference_array = np.absolute(image[:, axis]-threshold)
		index = difference_array.argmin()

		headNode = KdNode(image.mean(axis=0), axis, image_cur)
		cur_node = headNode

		heap_value = (slice_queue_instance[0], (slice_queue_instance[1], headNode))
		hq.heappush(slice_heap, heap_value)

		for i in range(slices):

			cur_node = hq.heappop(slice_heap)
			print("CURNODE", cur_node)
			(variance, ((axis, threshold) , cur_kd_node)) = cur_node

		
			image = cur_kd_node.image_view[cur_kd_node.image_view[:, axis].argsort()]
			
			self.axis_list.append(axis)
			

			left_array_arg = image[image[:, axis] < threshold]
			right_array_arg = image[image[:, axis] >= threshold]

			
			for idx, sub_array in enumerate([left_array_arg, right_array_arg]):
				if len(sub_array) > 0:
					slice_queue_instance_sub = self.get_slices(sub_array)
					if not (slice_queue_instance_sub is  None):
						axis = slice_queue_instance_sub[1][0]
						image = sub_array[sub_array[:, axis].argsort()]

						threshold = slice_queue_instance_sub[1][1]

						difference_array = np.absolute(image[:, axis]-threshold)
						index = difference_array.argmin()
						node_value = image[index]
						mean = image.mean(axis=0)

						subNode = KdNode(mean, axis, sub_array)
						self.median_list.append((subNode.value, i))

						if idx == 0:
							cur_kd_node.set_left(subNode)
						else:
							cur_kd_node.set_right(subNode)

						heap_value = (-slice_queue_instance_sub[0], (slice_queue_instance_sub[1], subNode))
						hq.heappush(slice_heap, heap_value)



		
		return headNode


	def find_mapping_value(self, value):
		depth_limit = self.slices
		cur_node = self.kd_tree
		#value = None
		dimension = 0
		ret = None
		pixel_values_ret = [0, 0, 0]

		while not cur_node is None:
			cur_val = cur_node.value
			axis = cur_node.axis
			left = cur_node.get_left()
			right = cur_node.get_right()
			ret = cur_node
			pixel_values_ret = cur_val

			if value[axis] < cur_val[axis]:
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
	kd_tree = KdTree(quantize_input, 24)
	#kd_tree.visualize_kd_tree()
	# m_list = [m for m, d in kd_tree.median_list]
	# m_list = np.vstack(m_list)
	# print(m_list)
	
	m_list = [m[1][1].value for m in kd_tree.slice_heap]
	m_list = np.vstack(m_list)
	print(m_list)
	#print(m_list - np.array([0, 125, 239]))
	#print(kd_tree.find_mapping_value([0, 125, 239]))
	quantize_output = np.zeros(quantize_input.shape)
	print("input", quantize_input[350,250])
	print("output", kd_tree.find_mapping_value(quantize_input[350,250]))
	print("input", quantize_input[190,190])
	print("output", kd_tree.find_mapping_value(quantize_input[190,190]))

	for i in range(quantize_input.shape[0]):
		for j in range(quantize_input.shape[1]):
			quantize_output[i, j] = kd_tree.find_mapping_value(quantize_input[i,j])
			#pass
	
	quantize_output = quantize_output.astype('uint8') 
	#quantize_output = cv2.cvtColor(quantize_output, cv2.COLOR_BGR2RGB)
	print(quantize_input[350,250])
	print("Quantized")
	print(quantize_output[350,250])
	# plt.imshow(quantize_output)
	# plt.show()
	cv2.imshow("quant", quantize_output)
	cv2.waitKey(0)

	plt.show()
if __name__ == '__main__':

	main()