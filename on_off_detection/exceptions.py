
class NoHistogramMinException(Exception):
	def __init__(self, message='No local minimum in count histogram. Can not compute count threshold'):
		super(NoHistogramMinException, self).__init__(message)

