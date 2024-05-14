import numpy as np
import math
from scipy import linalg
from scipy.sparse import diags
from typing import Union, Type, List


class Operators:
	"""
	Operators class to represent operators in the fock space based on numpy arrays
	"""

	def __init__(self, operator: np.array):
		"""
		:param operator: Matrix representing the operator in the fock space
		"""
		self.operator = operator
		self.assert_operator()
		self.n = operator.shape

	def assert_operator(self):
		assert len(self.operator.shape) == 2, "Operator must be a 2D matrix"

	def __repr__(self):
		return f"Operator is : \n ({self.operator})"

	@property
	def shape(self):
		return self.operator.shape

	@staticmethod
	def identity(n: int):
		"""
		Build the identity operator
		:param n: Dimension of the operator in the fock space
		:return: Identity Operators
		"""
		return Operators(np.eye(n))

	@staticmethod
	def annihilation(n: int):
		"""
		Build the annihilation operator
		:param n: Dimension of the operator in the fock space
		:return: Annihilation Operators
		"""
		return Operators(np.diag([np.sqrt(i) for i in range(1, n)], k=1))

	@staticmethod
	def creation(n: int):
		"""
		Build the creation operator
		:param n: Dimension of the operator in the fock space
		:return: Creation Operators
		"""
		return Operators(np.diag([np.sqrt(i) for i in range(1, n)], k=-1))

	@staticmethod
	def squeezed(n: int, spdc_emission: float):
		a = Operators.annihilation(n)
		a_dagger = Operators.creation(n)
		spdc_epsilon = np.arcsinh(np.sqrt(spdc_emission))
		argument = (a@a) + (a_dagger@a_dagger)
		argument_complex = -1j*argument*spdc_epsilon
		squeezed_operator = linalg.expm(argument_complex.operator)
		return Operators(squeezed_operator)

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
			elif res.shape[0] == 1 or res.shape[1] == 1:
				return State(res)
			else:
				raise ValueError("Invalid multiplication : the outcome is not a vector or a squared matrix")

	def __rmul__(self, other):
		if isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
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

	def __matmul__(self, other):
		"""
		Perform the kronecker product between two operators
		:param other: Operator to perform the tensor product with
		:return: Tensor product of the two operators
		"""
		if isinstance(other, Operators):
			res = np.kron(self.operator, other.operator)
			if res.shape[0] == 1 or res.shape[1] == 1:
				return State(res)
			else:
				return Operators(res)
		elif isinstance(other, State):
			res = np.kron(self.operator, other.state)
			if res.shape[0] == 1 or res.shape[1] == 1:
				return State(res)
			else:
				return Operators(res)
		else:
			raise NotImplementedError("Only Kronecker product between operators is implemented")

	def __rmatmul__(self, other):
		if isinstance(other, Operators):
			res = np.kron(other.operator, self.operator)
			if res.shape[0] == 1 or res.shape[1] == 1:
				return State(res)
			else:
				return Operators(res)
		elif isinstance(other, State):
			res = np.kron(other.state, self.operator)
			if res.shape[0] == 1 or res.shape[1] == 1:
				return State(res)
			elif res.shape[0] == res.shape[1]:
				return Operators(res)
		else:
			raise NotImplementedError("Only Kronecker product between operators is implemented")

	def __getitem__(self, item):
		return self.operator[item]

	def __setitem__(self, key, value):
		self.operator[key] = value

	@property
	def type(self):
		return "Operator"

	@staticmethod
	def bucket_detector(n: int, efficiency: float, noise: float):
		"""
		Build the bucket detector operator for APD detection
		:param n: Dimension of the operator in the fock space
		:param efficiency: Detection efficiency
		:param noise: Noise in the detector
		:return: Bucket detector operator
		"""
		observable = np.zeros((n, n))
		for i in range(n):
			observable[i, i] = 1 - ((1 - efficiency) ** i)
		observable = Operators(observable) + Operators.identity(n) * noise
		return observable

	@staticmethod
	def displacement(n: int, alpha: complex):
		"""
		Build the displacement operator
		:param n: Dimension of the operator in the fock space
		:param alpha: Complex number representing the displacement
		:return: Displacement operator
		"""
		argument = alpha * Operators.creation(n) - np.conj(alpha) * Operators.annihilation(n)
		return Operators(linalg.expm(argument.operator))

	@property
	def conj(self):
		"""
		Compute the conjugate of the operator
		"""
		return Operators(np.conj(self.operator))

	def dagger(self):
		"""
		Compute the conjugate transpose of the operator
		"""
		return Operators(self.operator.conj().T)

	def expect(self, state):
		"""
		Compute the expectation value of the operator in the given state
		:param state: State object to compute the expected value. Must be a column vector!
		:return: Expected value as a float
		"""
		my_operator = self.operator
		res = state.dagger() * Operators(my_operator) * state
		return res.state[0][0]


class State:
	"""
	State class to represent states in the fock space based on numpy arrays
	"""

	def __init__(self, state: np.array):
		"""
		:param state: Vector representing the state in the fock space
		"""
		self.state = state
		self.assert_state_vector()
		self.dimension = max(state.shape)

	def assert_state_vector(self):
		assert len(self.state.shape) == 2, "State must be a 2D vectors"

	@property
	def n(self):
		return self.dimension

	@staticmethod
	def vacuum(n: int):
		"""
		Built the vacuum state
		:param n: Dimension of the state in the fock space
		:return: Vacuum State
		"""
		return State(np.array([1] + [0] * (n - 1)).reshape(-1, 1))

	@staticmethod
	def one_photon(n: int):
		"""
		Built the one photon state
		:param n: Dimension of the state in the fock space
		:return: One photon State
		"""
		assert n > 0, "n must be greater than 0"
		assert n > 1, "n must be greater than 1"
		return State(np.array([0] + [1] + [0] * (n - 2)).reshape(-1, 1))

	@staticmethod
	def two_photon(n: int):
		"""
		Built the two photon state
		:param n: Dimension of the state in the fock space
		:return: Two photon State
		"""
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
		"""
		Compute the conjugate of the state
		:return:
		"""
		return State(np.conj(self.state))

	def dagger(self):
		"""
		Compute the conjugate transpose of the state
		:return: State
		"""
		return State(self.state.conj().T)

	def __add__(self, other):
		if isinstance(other, State):
			return State(self.state + other.state)
		else:
			raise NotImplementedError("Only addition between states is implemented")

	def __sub__(self):
		pass

	def __mul__(self, other):
		if isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
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

	def __rmul__(self, other):
		if isinstance(other, float) or isinstance(other, int):
			return State(self.state * other)
		else:
			raise NotImplementedError("Only right multiplication by scalar is implemented")

	def __matmul__(self, other):
		# Apply the kronecker product between two states
		if isinstance(other, State):
			res = np.kron(self.state, other.state)
			if res.shape[0] == 1 or res.shape[1] == 1:
				return State(res)
			else:
				return Operators(res)
		elif isinstance(other, Operators):
			res = np.kron(self.state, other.operator)
			if res.shape[0] == 1 or res.shape[1] == 1:
				return State(res)
			else:
				return Operators(res)
		else:
			raise NotImplementedError("Only Kronecker product between states and operators is implemented")

	def __rmatmul__(self, other):
		# Apply the kronecker product between two states
		if isinstance(other, State):
			res = np.kron(other.state, self.state)
			if res.shape[0] == 1 or res.shape[1] == 1:
				return State(res)
			else:
				return Operators(res)
		elif isinstance(other, Operators):
			res = np.kron(other.operator, self.state)
			if res.shape[0] == 1 or res.shape[1] == 1:
				return State(res)
			else:
				return Operators(res)
		else:
			raise NotImplementedError("Only Kronecker product between states is implemented")

	def __getitem__(self, item):
		return self.state[item]

	def __setitem__(self, key, value):
		self.state[key] = value