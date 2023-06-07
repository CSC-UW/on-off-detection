"""All exceptions for all on/off detection methods."""


# threshold
class NoHistogramMinException(Exception):
	def __init__(self, message='No local minimum in count histogram. Can not compute count threshold'):
		super().__init__(message)

# Hmmem
class NumericalErrorException(Exception):
	def __init__(self, message=None):
		super().__init__(message)

#All active or inactive init classification:
class FailedInitializationException(Exception):
	def __init__(self, message='All periods were on. Cannot train GLM.'):
		super().__init__(message)

ALL_METHOD_EXCEPTIONS = (
	NoHistogramMinException,
	NumericalErrorException,
	FailedInitializationException,
)