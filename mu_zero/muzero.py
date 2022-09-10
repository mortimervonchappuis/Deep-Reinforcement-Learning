from multiprocessing import Manager, Pipe, Process
from agent import Agent
from server import NetworkServer
from model import MuZeroNet
from replaybuffer import ReplayBuffer


class MuZero:
	def __init__(self, depth, env, gamma=1.0):
		self.depth = depth
		self.env   = env
		self.gamma = gamma


	def __call__(self, 
				 temperatur, 
				 n_process, 
				 epochs, 
				 path=None, 
				 checkpoint=100, 
				 update=1, 
				 rb_size=10_000,
				 K=5):
		server_cons = []
		processes   = []
		with Manager() as manager:
			replaybuffer = ReplayBuffer(rb_size, manager, K)
			for PID in range(n_process):
				server_con, agent_con = Pipe()
				server_cons.append(server_con)
				agent = Agent(self.env, 
							  agent_con, 
							  self.gamma, 
							  replaybuffer, 
							  self.depth,
							  self.env.action_mapping,
							  temperatur)
				proc = Process(target=agent.run, args=(epochs, PID))
				processes.append(proc)
			server = NetworkServer(MuZeroNet, server_cons, replaybuffer)
			server_args = (epochs, self.depth, update, checkpoint, path)
			server_proc = Process(target=server, args=server_args)
			# START PROCESSES
			for proc in processes:
				proc.start()
			server_proc.start()
			for proc in processes:
				proc.join()
			server_proc.join()



if __name__ == '__main__':
	from env import Chess
	mu = MuZero(5, Chess)
	mu({float('inf'): 1.0}, 2, 10)