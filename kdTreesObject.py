import numpy as np
import cv2
import matplotlib.pyplot as plt
from webcolors import rgb_to_hex

#https://www.baeldung.com/cs/k-d-trees#:~:text=The%20process%20starts%20by%20selecting,relative%20to%20the%20splitting%20hyperplane.
#https://www.ri.cmu.edu/pub_files/pub1/moore_andrew_1991_1/moore_andrew_1991_1.pdf
class KdNode(object):
	"""docstring for KdTree"""
	def __init__(self, value):
		super(KdNode, self).__init__()
		self.value = value
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
		self.image = image.reshape(-1, 3)
		self.depth_limit = depth_limit
		self.median_list = []
		self.axis_list = []

		self.kd_tree = self.create_kd_tree(self.image, self.depth_limit)

	def create_kd_tree(self, image, depth_limit):
		headNode = None

		if depth_limit <= 0:
			return None

		if image is None:
			# Would rather have an error message
			return None

		# if len(image.shape) != 3:
		# 	return None

		# if image.shape[2] != 3:
		# 	return None 

		#image = image.reshape(-1, 3)

		axis = (self.depth_limit - depth_limit) % 3

		self.axis_list.append(axis)

		sortted_array = image[image[:, axis].argsort()]
		
	
		median_index = len(sortted_array)//2
		median = image[median_index]
		self.median_list.append((median, depth_limit))
		cur_node = KdNode(median)
		#print(median)

		cur_node.set_left(self.create_kd_tree(image[0:median_index], depth_limit-1))
		cur_node.set_right(self.create_kd_tree(image[median_index:], depth_limit-1))

		return cur_node

	def visualize_kd_tree(self, hsv=False):
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

	def find_mapping_value(self, value):
		depth_limit = self.depth_limit
		cur_node = self.kd_tree
		#value = None
		dimension = 0
		ret = None
		while not cur_node is None:
			cur_val = cur_node.value
			ret = cur_node
			left = cur_node.get_left()
			right = cur_node.get_right()

			if left is None:
				pass
			if right is None:
				pass
			if value[dimension % 3] < cur_val[dimension % 3]:
				cur_node = left
			else:
				cur_node = right


		return ret.value


def main():
	quantize_input = cv2.imread("quantize_input.png")
	kd_tree = KdTree(quantize_input, 15)
	#kd_tree.visualize_kd_tree()
	m_list = [m for m, d in kd_tree.median_list]
	m_list = np.vstack(m_list)
	#print(m_list)
	#print(m_list - np.array([0, 125, 239]))
	#print(kd_tree.find_mapping_value([0, 125, 239]))
	quantize_output = np.zeros(quantize_input.shape)
	for i in range(quantize_input.shape[0]):
		for j in range(quantize_input.shape[1]):
			quantize_output[i, j] = kd_tree.find_mapping_value(quantize_input[i,j])
	
	quantize_output = quantize_output.astype('uint8') 
	#quantize_output = cv2.cvtColor(quantize_output.astype('uint8'), cv2.COLOR_BGR2RGB)

	print(quantize_output)
	cv2.imshow("quant", quantize_output)
	cv2.waitKey(0)

	plt.show()
if __name__ == '__main__':

	main()