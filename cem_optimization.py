import os

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from functools import partial
import numpy as np
import time

from trajectory_sampler import TrajSampler

import mujoco
import mujoco.mjx as mjx 
import jax
import jax.numpy as jnp


class cem_optimization():

	def __init__(self, num_dof=6, num_batch=100, num_steps=200, timestep=0.02, maxiter_cem=20, num_elite=0.1, 
			  eef_to_obj=2, obj_to_goal=0.03, push_align=0.1, obj_rot=0.1, eef_rot=0.1, collision=0.1):
		super(cem_optimization, self).__init__()

		self.key= jax.random.PRNGKey(0)
	 
		self.num_dof = num_dof
		self.num_batch = num_batch
		self.t = timestep
		self.num = num_steps
		self.num_elite = num_elite
		self.maxiter_cem = maxiter_cem
		self.ellite_num = int(self.num_elite*self.num_batch)
		self.t_fin = self.num*self.t
		self.cost_weights = {
			'eef_to_obj': eef_to_obj,
			'obj_to_goal': obj_to_goal,
			'push_align': push_align,
			'obj_rot': obj_rot,
			'eef_rot':eef_rot,
			'collision': collision,
		}

		self.sampler = TrajSampler(t_fin=self.t_fin, num=self.num, num_batch=self.num_batch, num_dof=self.num_dof)
		self.nvar = self.sampler.nvar

		self.alpha_mean = 0.6
		self.alpha_cov = 0.6

		self.lamda = 10
		self.g = 10
		self.vec_product = jax.jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

		self.model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 
		self.model = mujoco.MjModel.from_xml_path(self.model_path)
		self.data = mujoco.MjData(self.model)
		self.model.opt.timestep = self.t

		self.mjx_model = mjx.put_model(self.model)
		self.mjx_data = mjx.put_data(self.model, self.data)
		self.mjx_data = jax.jit(mjx.forward)(self.mjx_model, self.mjx_data)
		self.jit_step = jax.jit(mjx.step)

		self.geom_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'robot_{i}') for i in range(10)])
		target_geom_id = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'target_0')])
		target_mask = ~jnp.any(jnp.isin(self.mjx_data.contact.geom, target_geom_id), axis=1)
		self.mask = jnp.any(jnp.isin(self.mjx_data.contact.geom, self.geom_ids), axis=1)
		self.mask = jnp.logical_and(target_mask, self.mask)

		self.hande_id = self.model.body(name="hande").id
		self.tcp_id = self.model.site(name="tcp").id
		self.target_id = self.model.body(name="target_0").id
		self.target_qpos_idx = self.mjx_model.body_dofadr[self.target_id]

		self.compute_rollout_batch = jax.vmap(self.compute_rollout_single, in_axes = (0, None, None, None, None))
		self.compute_cost_batch = jax.vmap(self.compute_cost_single, in_axes = (0))

		self.print_info()


	def print_info(self):
		print(
			f'\n Default backend: {jax.default_backend()}'
			f'\n Model path: {self.model_path}',
			f'\n Timestep: {self.t}',
			f'\n CEM Iter: {self.maxiter_cem}',
			f'\n Number of batches: {self.num_batch}',
			f'\n Number of steps per trajectory: {self.num}',
			f'\n Time per trajectory: {self.t_fin}',
		)
	
	@partial(jax.jit, static_argnums=(0,))
	def mjx_step(self, mjx_data, thetadot_single):

		qvel = mjx_data.qvel.at[:self.num_dof].set(thetadot_single)
		mjx_data = mjx_data.replace(qvel=qvel)
		mjx_data = self.jit_step(self.mjx_model, mjx_data)

		theta = mjx_data.qpos[:self.num_dof]
		eef_rot = mjx_data.xquat[self.hande_id]	
		eef_pos = mjx_data.site_xpos[self.tcp_id]
		collision = mjx_data.contact.dist[self.mask]
		target_0_rot = mjx_data.xquat[self.target_id]	
		target_0_pos = mjx_data.xpos[self.target_id]

		return mjx_data, (theta, eef_pos, eef_rot, target_0_pos, target_0_rot, collision)

	@partial(jax.jit, static_argnums=(0,))
	def compute_rollout_single(self, thetadot, init_pos, init_vel, init_target_pos, init_target_rot):
	
		mjx_data = self.mjx_data
		qvel = mjx_data.qvel.at[:self.num_dof].set(init_vel)
		qpos = mjx_data.qpos.at[:self.num_dof].set(init_pos)
		qpos = qpos.at[self.target_qpos_idx : self.target_qpos_idx + 3].set(init_target_pos)
		qpos = qpos.at[self.target_qpos_idx + 3 : self.target_qpos_idx + 7].set(init_target_rot)
		mjx_data = mjx_data.replace(qvel=qvel, qpos=qpos)

		thetadot_single = thetadot.reshape(self.num_dof, self.num)
		mjx_data, out = jax.lax.scan(self.mjx_step, mjx_data, thetadot_single.T, length=self.num)
		theta, eef_pos, eef_rot, target_0_pos, target_0_rot, collision = out
		return theta.T.flatten(), eef_pos, eef_rot, target_0_pos, target_0_rot, collision
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_cost_single(self, thetadot, eef_pos, eef_rot, target_0_pos, target_0_rot, collision, target_pos, target_rot):
		eef_obj = eef_pos[:, :-1] - target_0_pos[:, :-1]
		obj_goal = target_0_pos[:, :-1] - target_pos[:-1]

		eef_obj_dist = jnp.linalg.norm(eef_obj, axis=1)
		obj_goal_dist = jnp.linalg.norm(obj_goal, axis=1)

		push_align = jnp.abs(jnp.sum(eef_obj*obj_goal, axis=1)/(eef_obj_dist * obj_goal_dist) - 1)

		approach_offset = (target_0_pos - target_pos)
		approach_offset = approach_offset/(jnp.abs(approach_offset)+0.001)*0.05
		approach_offset = approach_offset.at[:, 2].set(0.01)
		eef_obj_dist = jnp.linalg.norm(eef_pos - (target_0_pos+approach_offset), axis=1)

		dot_product = jnp.abs(jnp.dot(target_0_rot/jnp.linalg.norm(target_0_rot, axis=1).reshape(1, self.num).T, target_rot/jnp.linalg.norm(target_rot)))
		dot_product = jnp.clip(dot_product, -1.0, 1.0)
		obj_cost_r_ = 2 * jnp.arccos(dot_product)
		obj_cost_r = jnp.sum(obj_cost_r_)

		eef_rot_target = jnp.array([0, 0.70711, 0.70711, 0])
		dot_product = jnp.abs(jnp.dot(eef_rot/jnp.linalg.norm(eef_rot, axis=1).reshape(1, self.num).T, eef_rot_target/jnp.linalg.norm(eef_rot_target)))
		dot_product = jnp.clip(dot_product, -1.0, 1.0)
		eef_cost_r_ = 2 * jnp.arccos(dot_product)
		eef_cost_r = jnp.sum(eef_cost_r_)

		y = 0.8
		collision = collision.T
		g = -collision[:, 1:]+collision[:, :-1]-y*collision[:, :-1]
		cost_c = jnp.sum(jnp.max(g.reshape(g.shape[0], g.shape[1], 1), axis=-1, initial=0)) + jnp.sum(jnp.where(collision<0, True, False))

		cost = (
            self.cost_weights["eef_to_obj"] * jnp.sum(eef_obj_dist)
            + self.cost_weights["obj_to_goal"] * jnp.sum(obj_goal_dist)
            + self.cost_weights["obj_rot"] * obj_cost_r
			+ self.cost_weights["eef_rot"] * eef_cost_r
            + self.cost_weights["push_align"] * jnp.sum(push_align)
            + self.cost_weights["collision"] * cost_c
        )
		return cost, cost_c
	
	@partial(jax.jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, xi_filtered):
		idx_ellite = jnp.argsort(cost_batch)
		cost_ellite = cost_batch[idx_ellite[0:self.ellite_num]]
		xi_ellite = xi_filtered[idx_ellite[0:self.ellite_num]]
		return xi_ellite, idx_ellite, cost_ellite
	
	@partial(jax.jit, static_argnums=(0,))
	def comp_prod(self, diffs, d ):
		term_1 = jnp.expand_dims(diffs, axis = 1)
		term_2 = jnp.expand_dims(diffs, axis = 0)
		prods = d * jnp.outer(term_1,term_2)
		return prods	
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_mean_cov(self, cost_ellite, mean_control_prev, cov_control_prev, xi_ellite):
		w = cost_ellite
		w_min = jnp.min(cost_ellite)
		w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
		sum_w = jnp.sum(w, axis = 0)
		mean_control = (1-self.alpha_mean)*mean_control_prev + self.alpha_mean*(jnp.sum( (xi_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
		diffs = (xi_ellite - mean_control)
		prod_result = self.vec_product(diffs, w)
		cov_control = (1-self.alpha_cov)*cov_control_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.0001*jnp.identity(self.nvar)
		return mean_control, cov_control
	
	@partial(jax.jit, static_argnums=(0,))
	def cem_iter(self, carry, _):
		init_pos, init_vel, init_target_pos, init_target_rot, target_pos, target_rot, xi_mean, xi_cov, key, state_term = carry

		xi_mean_prev = xi_mean 
		xi_cov_prev = xi_cov

		thetadot, xi_samples, key = self.sampler.generate_samples(key=key, xi_mean=xi_mean, xi_cov=xi_cov, state_term=state_term)

		theta, eef_pos, eef_rot, target_0_pos, target_0_rot, collision = self.compute_rollout_batch(thetadot, init_pos, init_vel, init_target_pos, init_target_rot)
		cost_batch, cost_c_batch = self.compute_cost_batch(thetadot, eef_pos, eef_rot, target_0_pos, target_0_rot, collision, target_pos, target_rot)

		xi_ellite, idx_ellite, cost_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
		xi_mean, xi_cov = self.compute_mean_cov(cost_ellite, xi_mean_prev, xi_cov_prev, xi_ellite)

		carry = (init_pos, init_vel, init_target_pos, init_target_rot, target_pos, target_rot, xi_mean, xi_cov, key, state_term)
		return carry, (cost_batch, cost_c_batch, thetadot, theta)

	@partial(jax.jit, static_argnums=(0,))
	def compute_cem(
		self, xi_mean, 
		init_pos=jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0]), 
		init_vel=jnp.zeros(6), 
		init_acc=jnp.zeros(6),
		target_pos=jnp.zeros(3),
		target_rot=jnp.zeros(4),
		init_target_pos=jnp.zeros(3), 
		init_target_rot=jnp.zeros(4)
		):

		target_pos = jnp.tile(target_pos, (self.num_batch, 1))
		target_rot = jnp.tile(target_rot, (self.num_batch, 1))

		theta_init = jnp.tile(init_pos, (self.num_batch, 1))
		thetadot_init = jnp.tile(init_vel, (self.num_batch, 1))
		thetaddot_init = jnp.tile(init_acc, (self.num_batch, 1))
		thetadot_fin = jnp.zeros((self.num_batch, self.num_dof))
		thetaddot_fin = jnp.zeros((self.num_batch, self.num_dof))

		state_term = jnp.hstack((theta_init, thetadot_init, thetaddot_init, thetadot_fin, thetaddot_fin))
		state_term = jnp.asarray(state_term)
		
		xi_cov = 10*jnp.identity(self.nvar)
  
		key, subkey = jax.random.split(self.key)

		carry = (init_pos, init_vel, init_target_pos, init_target_rot, target_pos, target_rot, xi_mean, xi_cov, key, state_term)
		scan_over = jnp.array([0]*self.maxiter_cem)
		carry, out = jax.lax.scan(self.cem_iter, carry, scan_over, length=self.maxiter_cem)
		cost_batch, cost_c_batch, thetadot, theta = out

		idx_min = jnp.argmin(cost_batch[-1])
		cost = jnp.min(cost_batch, axis=1)
		thetadot = thetadot[-1][idx_min].reshape((self.num_dof, self.num)).T
		theta = theta[-1][idx_min].reshape((self.num_dof, self.num)).T
		cost_c = cost_c_batch[-1][idx_min]
		xi_mean = carry[6]

		return cost, cost_c, thetadot, theta, xi_mean
	
def main():

	start_time = time.time()
	opt_class = cem_optimization(num_dof=6, num_batch=2000, num_steps=50, maxiter_cem=30,
                           w_pos=1, w_rot=0.5, w_col=10, num_elite=0.05, timestep=0.05)

	start_time_comp_cem = time.time()
	target_pos = np.array([-0.3, 0, 0.9])
	target_rot = np.array([0, 0.70711, 0.70711, 0])
	cost, cost_g, cost_r, cost_c, thetadot, theta, xi_mean = opt_class.compute_cem(xi_mean=np.zeros(opt_class.nvar), target_pos=target_pos, target_rot=target_rot)

	print(f"Total time: {round(time.time()-start_time, 2)}s")
	print(f"Compute CEM time: {round(time.time()-start_time_comp_cem, 2)}s")

	np.savetxt('data/costs.csv',cost, delimiter=",")
	np.savetxt('data/thetadot.csv',thetadot, delimiter=",")
	np.savetxt('data/theta.csv',theta, delimiter=",")
	np.savetxt('data/cost_g.csv',cost_g, delimiter=",")
	np.savetxt('data/cost_r.csv',cost_r, delimiter=",")
	np.savetxt('data/cost_c.csv',[cost_c], delimiter=",")

	
	
if __name__ == "__main__":
	main()


  	
