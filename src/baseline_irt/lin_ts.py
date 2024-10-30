import numpy as np
import random

class LinearTS:
	def __init__(self, state_dims:int, action_size:int, params:dict):
		self.state_dims = state_dims
		self.action_size = action_size

		self.event_buffer = []

		self.A = [np.eye(self.state_dims) for _ in range(self.action_size)]
		self.b = [np.zeros((self.state_dims, )) for _ in range(self.action_size)]

		self.n_queried_actions = 0

	def getName(self):
		return "linear TS"
	
	def queryContexts(self, ctx:list) -> int:
		ctx = [np.array(x) for x in ctx]
		est_rs = []

		for a in range(self.action_size):
			A_inv = np.linalg.inv(self.A[a])
			theta_hat = A_inv.dot(self.b[a])
			theta_sampled = np.random.multivariate_normal(theta_hat, A_inv)
			est_rs.append(theta_sampled.dot(ctx[a]))
			
		bandit_a = np.argmax(est_rs)

		return bandit_a

	def updateContext(self, ctx:np.array, a:int, r:float):
		ctx = np.array(ctx[a])

		self.A[a] += np.outer(ctx, ctx.T)
		self.b[a] += r * ctx