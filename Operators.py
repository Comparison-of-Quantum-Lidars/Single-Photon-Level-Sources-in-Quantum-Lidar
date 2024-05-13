import numpy as np
import math
from scipy import linalg
from typing import Union, Type, List


class Operators:

	def __init__(self, dimension):
		self.operator = np.zeros((dimension, dimension))
		self.n = dimension

	def __add__(self, other):
		if isinstance(other, Operators):
			assert self.n == other.n, "Operators must have the same dimension"
			return self.operator + other.operator
		elif isinstance(other, State):
			pass

	def __repr__(self):
		return f"Operator is ({self.operator})"

	@property
	def shape(self):
		return self.operator.shape


class State:

	def __init__(self, state: np.array):
		self.state = state
		self.assert_state_vector()
		self.dimension = state.shape[0]

	def assert_state_vector(self):
		assert len(self.state.shape) == 2, "State must be a 2D vectors"
		assert self.state.shape[1] == 1, "State must be a column vector"
		assert math.isclose(linalg.norm(self.state), 1, rel_tol=1e-6), "State must be normalized"

	@property
	def n(self):
		return self.dimension

	@staticmethod
	def vacuum(n: int):
		return State(np.array([1] + [0] * (n - 1)).reshape(-1, 1))

	@staticmethod
	def one_photon(n: int):
		assert n > 0, "n must be greater than 0"
		assert n > 1, "n must be greater than 1"
		return State(np.array([0] + [1] + [0] * (n - 2)).reshape(-1, 1))

	@staticmethod
	def two_photon(n: int):
		assert n > 0, "n must be greater than 0"
		assert n > 2, "n must be greater than 2"
		return State(np.array([0] * 2 + [1] + [0] * (n - 3)).reshape(-1, 1))

	@property
	def type(self):
		return "State"

	def __repr__(self):
		return f"State is ({self.state})"

	@property
	def conj(self):
		return State(np.conj(self.state))

	def dagger(self):
		"""
		Compute the conjugate transpose of the state
		:return: State
		"""
		return State(self.state.conj().T)

	def __add__(self, other):
		pass

	def __sub__(self):
		pass

	def __mul__(self):
		pass


class DisplacementOperator(Operators):

	def __init__(self):
		super().__init__()

	def __add__(self, other: [Operators, State]):
		pass

	def __sub__(self, other: [Operators, State]):
		pass

	def __mul__(self, other: [Operators, State]):
		pass

	def expected_value(self, state: State):
		pass


class AnnihilationOperator(Operators):

	def __init__(self, n: int):
		super().__init__(n)
		self.n = n


class CreationOperator(Operators):

	def __init__(self, n: int):
		super().__init__(n)


class BucketDetector(Operators):

	def __init__(self, n: int):
		super(BucketDetector, self).__init__(n)

class Identity(Operators):

	def __init__(self, n: int):
		super(Identity, self).__init__(n)
