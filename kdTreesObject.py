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
		return self.left

	def get_right(self):
		return self.right

	def set_left(self, left):
		self._left = left

	def set_right(self, right):
		self._right = right

		


class KdTree(object):
	"""docstring for KdTree"""
	def __init__(self, image, depth):
		super(KdTree, self).__init__()
		self.image = image.reshape(-1, 3)
		self.depth = depth
		self.median_list = []
		self.axis_list = []

		self.kd_tree = self.create_kd_tree(self.image, self.depth)

	def create_kd_tree(self, image, depth):
		headNode = None

		if depth <= 0:
			return None

		if image is None:
			# Would rather have an error message
			return None

		# if len(image.shape) != 3:
		# 	return None

		# if image.shape[2] != 3:
		# 	return None 

		#image = image.reshape(-1, 3)

		axis = depth % 3

		self.axis_list.append(axis)

		sortted_array = image[image[:, axis].argsort()]
		
	
		median_index = len(sortted_array)//2
		median = image[median_index]
		self.median_list.append(median)
		cur_node = KdNode(median)
		#print(median)

		cur_node.set_left(self.create_kd_tree(image[0:median_index], depth-1))
		cur_node.set_right(self.create_kd_tree(image[median_index:], depth-1))

		return cur_node

	def visualize_kd_tree(self, hsv=False):
		image = self.image.reshape(-1, 3)
		colors = np.apply_along_axis(rgb_to_hex, -1, image)
		dim1 = image[:, 0]
		dim2 = image[:, 1]
		dim3 = image[:, 2]
		ax = plt.gca(projection="3d")
		#ax.scatter(dim1, dim2, dim3, c=colors)
		for idx, median in enumerate(self.median_list):
			index = self.axis_list[idx]

			axes_dim = np.linspace(0, 255, 255)
			static_dim1 = np.linspace(median[(index + 1) % 3], median[(index + 1) % 3], 255)
			static_dim2 = np.linspace(median[(index + 2) % 3], median[(index + 2) % 3], 255)
			

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
	print(kd_tree.kd_tree.value)
	kd_tree.visualize_kd_tree()

if __name__ == '__main__':

	main()