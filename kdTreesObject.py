
class KdNode(object):
	"""docstring for KdTree"""
	def __init__(self, value):
		super(KdNode, self).__init__()
		self.value = image
		self._left = None
		self._right = None

	def get_left():
		return self.left

	def get_right():
		return self.right

	def set_left(left):
		self._left = left

	def set_right(right):
		self._right = right

		


class KdTree(object):
	"""docstring for KdTree"""
	def __init__(self, image, depth):
		super(KdTree, self).__init__()
		self.image = image
		self.depth = self.depth
		self.kd_tree = self.create_kd_tree(self.image, self.num_nodes)

	def create_kd_tree(image, depth):
		headNode = None

		if depth <= 0:
			return None

		if image is None:
			# Would rather have an error message
			return None

		image = image.flatten()

		depth = 0
		axis = depth % len(image.size)

		
		sortted_array = np.sort(median, axis)

		cur_node = KdNode(median)
		median_index = len(sortted_array)//2
		median = np.median(image, axis)

		cur_node.set_left(KdTree(image[np.where(image[0:median_index])]), depth-1)
		cur_node.set_left(KdTree(image[np.where(image[median_index:])]), depth-1)

		return cur_node




