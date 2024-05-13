import numpy as np
import math
from scipy import linalg
from scipy.sparse import diags
from typing import Union, Type, List


class Operators:

	def __init__(self, operator: np.array):
		self.operator = operator
		self.assert_operator()
		self.n = operator.shape[0]

	def assert_operator(self):
		assert len(self.operator.shape) == 2, "Operator must be a 2D matrix"
		assert self.operator.shape[0] == self.operator.shape[1], "Operator must be a square matrix"

	def __repr__(self):
		return f"Operator is : \n ({self.operator})"

	@property
	def shape(self):
		return self.operator.shape

	@staticmethod
	def identity(n: int):
		return Operators(np.eye(n))

	@staticmethod
	def annihilation(n: int):
		return Operators(np.diag([np.sqrt(i) for i in range(1, n)], k=1))


	@staticmethod
	def creation(n: int):
		return Operators(np.diag([np.sqrt(i) for i in range(1, n)], k=-1))

	def __mul__(self, other):
		if isinstance(other, float) or isinstance(other, int):
			return Operators(self.operator * other)
		if isinstance(other, Operators):
			my_operator = self.operator
			other_operator = other.operator
			return Operators(np.matmul(my_operator, other_operator))
		if isinstance(other, State):
			res = np.matmul(self.operator, other.state)
			# check if matrix or vector
			if res.shape[0] == res.shape[1]:
				return Operators(res)
			elif res.shape[0]==1 or res.shape[1]==1:
				return State(res)
			else:
				raise ValueError("Invalid multiplication : the outcome is not a vector or a squared matrix")

	def __rmul__(self, other):
		if isinstance(other, float) or isinstance(other, int):
			return Operators(self.operator * other)
		if isinstance(other, Operators):
			my_operator = self.operator
			other_operator = other.operator
			return Operators(np.matmul(other_operator, my_operator))
		if isinstance(other, State):
			return State(np.matmul(self.operator, other.state))

	def __add__(self, other):
		if isinstance(other, Operators):
			return Operators(self.operator + other.operator)

	def __sub__(self, other):
		if isinstance(other, Operators):
			return Operators(self.operator - other.operator)

	def __eq__(self, other):
		if isinstance(other, Operators):
			return np.allclose(self.operator, other.operator)
		return False

	@property
	def type(self):
		return "Operator"

	@staticmethod
	def bucket_detector(n: int, efficiency: float, noise: float):
		observable = np.zeros((n, n))
		for i in range(n):
			observable[i, i] = 1 - ((1 - efficiency) ** i)
		observable = Operators(observable) + Operators.identity(n) * noise
		return observable

	@staticmethod
	def displacement(n: int, alpha: complex):
		argument = alpha*Operators.creation(n) - np.conj(alpha)*Operators.annihilation(n)
		return Operators(linalg.expm(argument.operator))

	@property
	def conj(self):
		return Operators(np.conj(self.operator))

	def dagger(self):
		return Operators(self.operator.conj().T)

	def expect(self, state):
		my_operator = self.operator
		res = state.dagger() * Operators(my_operator) * state
		return res.state[0][0]

class State:

	def __init__(self, state: np.array):
		self.state = state
		self.assert_state_vector()
		self.dimension = state.shape[0]

	def assert_state_vector(self):
		assert len(self.state.shape) == 2, "State must be a 2D vectors"

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

	def __mul__(self, other):
		if isinstance(other, float) or isinstance(other, int):
			return State(self.state * other)
		if isinstance(other, State):
			res = np.matmul(self.state, other.state)
			if res.shape[0] == 1:
				return State(res)
			else:
				raise ValueError("Invalid multiplication : the outcome is not a vector")
		if isinstance(other, Operators):
			res = np.matmul(self.state, other.operator)
			if res.shape[0] == 1:
				return State(res)
			else:
				raise ValueError("Invalid multiplication : the outcome is not a vector")

