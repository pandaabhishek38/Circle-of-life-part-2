import random
import ast
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap
figure(figsize=[8, 8], dpi = 100)
import math
import networkx
import copy


class Graph():
	num_nodes = 50
	nodes_list = []
	graph_dict = {}
	agent_pos = -1
	pred_pos = -1
	prey_pos = -1
	dist_dict = {}
	connectivity_matrix = np.array([[0 for i in range(50)] for j in range(50)])
	possible_starts = []

	def edge_available(self, current_node, check_list = []):
		edge_sum = 0
		graph_neighbours = []
		for i in range(2,6):
			graph_neighbours.append(self.nodes_list[current_node-i])
		for i in range(2,6):
			graph_neighbours.append(self.nodes_list[(current_node+i)%len(self.nodes_list)])
		print("graph_neighbours: " + str(graph_neighbours))
		for i in graph_neighbours:
			if (i != current_node):
				print("for key " + str(i))
				print("Adding length of " + str(self.graph_dict[self.nodes_list[i]]) + " to the sum")
				edge_sum += len(self.graph_dict[self.nodes_list[i]])
				if (len(self.graph_dict[self.nodes_list[i]]) < 3):
					check_list.append(self.nodes_list[i])
		##print(edge_sum)
		if edge_sum == 24:
			return False
		else:
			return True

	def add_edge_3(self, current_node):
		flag = False
		print("Current Node: " + str(current_node))
		print("Order of currrent node: " + str(len(self.graph_dict[self.nodes_list[current_node]])))
		print("Current Node Neighbours: " + str(self.graph_dict[current_node]))
		check_list = []
		if (self.edge_available(current_node, check_list)):
			while(flag == False and len(self.graph_dict[self.nodes_list[current_node]]) < 3):
				print("check_list: " + str(check_list))
				x = random.randint(0,len(check_list)-1)
				x = check_list[x]
				other_node = self.nodes_list[x]
				print("Node being considered: " + str(other_node))
				print("Order of that node: " + str(len(self.graph_dict[other_node])))
				print("That node Neighbours: " + str(self.graph_dict[other_node]))

				if ((other_node != self.nodes_list[current_node]) and other_node not in self.graph_dict[current_node]):
					self.graph_dict[current_node].append(other_node)
					self.graph_dict[other_node].append(current_node)
					print("Edge added")
					print(str(current_node) + ": " + str(self.graph_dict[current_node]))
					print(str(other_node) + ": " + str(self.graph_dict[other_node]))
					flag = True

	def create_graph(self):
		self.nodes_list = []
		for i in range(50):
			self.nodes_list.append(i)

		for i in range(len(self.nodes_list)):
			print("i:" + str(i) + " Current:" + str(self.nodes_list[i]))
			print(" and Previous: " + str(self.nodes_list[i-1]) + " and Next: " + str(self.nodes_list[(i+1)%len(self.nodes_list)]))
			self.graph_dict[self.nodes_list[i]] = [self.nodes_list[i-1],self.nodes_list[(i+1)%len(self.nodes_list)]]
		
		for i in range(len(self.nodes_list)):
			self.add_edge_3(i)

		##########CONNECTIVITY MATRIX START##########
		for i in range(len(self.connectivity_matrix)):
			for val in self.graph_dict[i]:
				self.connectivity_matrix[i][val] = 1
		##########CONNECTIVITY MATRIX END##########


	def initialize_positions(self):
		self.possible_starts = self.nodes_list.copy()
		self.agent_pos = random.randint(0,49)
		
		#Removing agents position from the list of possible nodes. So, the agent and the pred and the agent and the prey will never start from the same pos
		#However, the pred and the prey can spawn at the same node as told
		self.possible_starts.remove(self.agent_pos)
		self.pred_pos = random.choice(self.possible_starts)
		self.prey_pos = random.choice(self.possible_starts)


	def calc_path(self, graph_dict, from_pos, to_pos):
		closed_list = []
	
		queue = [[from_pos]]
	
		if from_pos == to_pos:
			return [from_pos]
	
		while queue:
			temp_path = queue.pop(0)
			node = temp_path[-1]
		
			if node not in closed_list:
				neighbours = graph_dict[node]
				
				for neighbour in neighbours:
					shortest_path = list(temp_path)
					shortest_path.append(neighbour)
					queue.append(shortest_path)
				
					if neighbour == to_pos:
						return shortest_path
				closed_list.append(node)

class Prey():
	def move_prey(self,graph_dict, prey_pos):
		possible_next = []
		for i in graph_dict[prey_pos]:
			possible_next.append(i)
		possible_next.append(prey_pos)
		print("Prey possible_next: " + str(possible_next))
		prey_pos = random.choice(possible_next)
		return prey_pos


class Predator():
	def move_pred(self, graph_dict, agent_pos, pred_pos):
		print("Predator Position: " + str(pred_pos))
		y = random.randint(0,10)
		if (y >= 4): #random.random() gives [0,1.0). So <= 40% would be [0,3.9999...]
			graph_1 = Graph()
			path = graph_1.calc_path(graph_dict, pred_pos, agent_pos)
			pred_pos = path[1]
		else:
			pred_pos = random.choice(graph_dict[pred_pos])
		print("Predators new position: " + str(pred_pos))
		return pred_pos

class agents_common():
	step_count = 1
	pred_probabilities = {}
	prey_probabilities = {}
	last_seen_prey = -1
	last_seen_Pred = -1
	transition_prob =  {}
	path_dict = {}
	points = []
	rewards = {}
	utility_vals = {}
	utility_updated = {}
	starting_pt = []
	marked = {}

	def decide_node_even(self, graph_dict, dist_to_prey, dist_to_pred, agent_pos, pred_pos, candidate_nodes):
		min_pred_dist = dist_to_pred[agent_pos]
		if min_pred_dist >= 6:
			agent_pos = self.decide_node(dist_to_prey, dist_to_pred, agent_pos, candidate_nodes)
		else:
			graph = Graph()
			#calculating possible next position of predator by taking shortest route from predator to agent. 
			#Distracted predator is not taken care of int this approach as it has lesser probability of happening
			#We observed during trail runs that the predator many times took the shortest path and not the distracted path. So not including
			path_from_Pred = graph.calc_path(graph_dict, pred_pos, agent_pos)
			if len(path_from_Pred) > 1:
				pred_next_node = path_from_Pred[1]
			else:
				pred_next_node = agent_pos
			min_pred_next_dist = len(graph.calc_path(graph_dict, pred_next_node, agent_pos))
			min_prey_dist = dist_to_prey[agent_pos]
			narrowed_candidates = []
			dist_to_pred_next = {}
			del dist_to_pred[agent_pos]
			del dist_to_prey[agent_pos]

			#Generating list of distances to the predators next possible position
			for val in graph_dict[agent_pos]:
				print("Calculating distance to pred("+str(pred_pos)+") from neighbour("+str(val)+") of current("+str(agent_pos)+")")
				dist_to_pred_next[val] = len(graph.calc_path(graph_dict, val, pred_pos))

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes Closer to Prey and Farther from Predator - Condition 1 - Start
			for i in candidate_nodes:
				#Shortlist all neighbouring nodes that are closer to prey and farther from the predator and predator_next
				if dist_to_prey[i] < min_prey_dist and dist_to_pred[i] > min_pred_dist and dist_to_pred_next[i] > min_pred_next_dist:
					narrowed_candidates.append(i)
				else:
					#remove key, val pairs from distance to prey/pred dictionaries for nodes not selected above
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					del lcl_dist_to_prey[i]
			
			print("Inside condition 1 check for even")

			#Checking if there are multiple nodes that are closer to prey and farther from the predator
			if (len(narrowed_candidates) > 1):
				narrow_pred_next = []
				narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]

				#Checking if there are multiple nodes that are closer to preds next possible node with the same distance
				if (len(narrow_pred_next) > 1):
					#remove key, val pairs from distance to predator dictionary which are not selected above
					for key in lcl_dist_to_pred_next.keys():
						if key not in narrow_pred_next:
							del lcl_dist_to_pred[key]

					#If multiple, selecting node with maximum distance to predator
					narrow_pred = []
					#If multiple, selecing node with maximum distance from predator
					narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]
					if (len(narrow_pred) > 1):
						for key in lcl_dist_to_pred.keys():
							if key not in narrow_pred:
								del lcl_dist_to_prey[key]

						narrow_prey = []
						narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]
						if (len(narrow_prey) > 1):
							return narrow_prey[random.randint(0,len(narrow_prey)-1)]
						elif (len(narrow_prey) == 1):
							#If still multiple after selecting min(distance from prey) and max(distance from predator), max(distance from predator next), selecting randomly
							return narrow_prey[0]
					elif (len(narrow_pred) == 1):
						return narrow_pred[0]
				elif (len(narrow_pred_next) == 1):
					return narrow_pred_next[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes Closer to Prey and Farther from Predator - Condition 1 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes Closer to Prey and Farther from Predators next pos - Condition 1_1 - Start
			for i in candidate_nodes:
				#Shortlist all neighbouring nodes that are closer to prey and farther from predator_next
				if dist_to_prey[i] < min_prey_dist and dist_to_pred_next[i] > min_pred_next_dist:
					narrowed_candidates.append(i)
				else:
					#remove key, val pairs from distance to prey, pred, pred_next dictionaries for nodes not selected above
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					del lcl_dist_to_prey[i]
			
			print("Inside condition 1_1 check for even")

			#Checking if there are multiple nodes that are closer to prey and farther from the predator next
			if (len(narrowed_candidates) > 1):
				narrow_pred_next = []
				narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]

				#Checking if there are multiple nodes that are farthest from pred next location
				if (len(narrow_pred_next) > 1):
					#remove key, val pairs from distance to prey dictionary which are not selected above
					for key in lcl_dist_to_pred_next.keys():
						if key not in narrow_pred_next:
							del lcl_dist_to_prey[key]

					#If multiple, selecting node with minimum distance from prey
					narrow_prey = []
					narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]
					if (len(narrow_prey) > 1):
						#If more than one with same distance, select at random
						return narrow_prey[random.randint(0,len(narrow_prey)-1)]
					elif (len(narrow_prey) == 1):
						return narrow_prey[0]
				elif (len(narrow_pred_next) == 1):
					return narrow_pred_next[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]

			#selecting nodes Closer to Prey and Farther from Predators next pos - Condition 1_1 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes Closer to Prey and Farther from Predators current position - Condition 1_2 - Start
			for i in candidate_nodes:
				#Shortlist all neighbouring nodes that are closer to prey and farther from the predator
				if dist_to_prey[i] < min_prey_dist and dist_to_pred[i] > min_pred_dist:
					narrowed_candidates.append(i)
				else:
					#remove key, val pairs from distance to prey,pred dictionaries for nodes not selected above
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					del lcl_dist_to_prey[i]
			
			print("Inside condition 1_2 check for even")

			#Checking if there are multiple nodes that are closer to prey and farther from the predator
			if (len(narrowed_candidates) > 1):
				narrow_pred = []
				narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]

				#Checking if there are multiple nodes that are farther from the predator with the same distance
				if (len(narrow_pred) > 1):
					#remove key, val pairs from distance to prey dictionary which are not selected above
					for key in lcl_dist_to_pred.keys():
						if key not in narrow_pred:
							del lcl_dist_to_prey[key]

					#If multiple, selecting node with minimum distance from prey
					narrow_prey = []
					narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]
					if (len(narrow_prey) > 1):
						#If still multiple that have same minimum distance from prey, choose randomly
						return narrow_prey[random.randint(0,len(narrow_prey)-1)]
					elif (len(narrow_prey) == 1):
						#If only 1 with minimum distance from prey, choose that
						return narrow_prey[0]
				#If only 1 with maximum distance to predator, choose that
				elif (len(narrow_pred) == 1):
					return narrow_pred[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes Closer to Prey and Farther from Predators current position - Condition 1_2 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes farther from Predator and predator_next - Condition 2 - Start
			for i in candidate_nodes:
				if dist_to_pred[i] > min_pred_dist and dist_to_pred_next[i] > min_pred_next_dist:
					narrowed_candidates.append(i)
				else:
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
					del lcl_dist_to_prey[i]
			
			print("Inside condition 2 check")

			#Checking if there are multiple nodes that are farther from the predator and predator_next positions
			if (len(narrowed_candidates) > 1):
				narrow_pred_next = []
				narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]

				#Checking if there are multiple nodes witht he same distance that are farthest from predators next possible location
				if (len(narrow_pred_next) > 1):
					#remove key, val pairs from distance to predator dictionary which are not selected above
					for key in lcl_dist_to_pred_next.keys():
						if key not in narrow_pred_next:
							del lcl_dist_to_pred[key]

					#If multiple, selecting node with maximum distance to predator
					narrow_pred = []
					narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]
					if (len(narrow_pred) > 1):
						#remove key, val pairs from distance to prey dictionary which are not selected above
						for key in lcl_dist_to_pred.keys():
							if key not in narrow_pred:
								del lcl_dist_to_prey[key]

						#Select nodes that are closes to the prey from the nodes shortlisted above
						narrow_prey = []
						narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]
						if (len(narrow_prey) > 1):
							#If multiple, choose randomly
							return narrow_prey[random.randint(0,len(narrow_prey)-1)]
						elif (len(narrow_prey) == 1):
							#If only one node, choose that
							return narrow_prey[0]
					elif (len(narrow_pred) == 1):
						return narrow_pred[0]
				elif (len(narrow_pred_next) == 1):
					return narrow_pred_next[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes farther from Predator and predator_next - Condition 2 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes farther from Predator and predator_next - Condition 2_1 - Start
			for i in candidate_nodes:
				if dist_to_pred_next[i] > min_pred_next_dist:
					narrowed_candidates.append(i)
				else:
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
			
			print("Inside condition 2_1 check")

			#Select the node that is farthest from teh predators next location
			if (len(narrowed_candidates) > 1):
				narrow_pred_next = []
				narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]

				#Checking if there are multiple nodes that farthest from pred next location with same distance
				if (len(narrow_pred_next) > 1):
					#remove key, val pairs from distance to predator dictionary which are not selected above
					for key in lcl_dist_to_pred_next.keys():
						if key not in narrow_pred_next:
							del lcl_dist_to_pred[key]

					narrow_pred = []
					#select node that is farthest from the predators current position
					narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]
					if (len(narrow_pred) > 1):
						#If multiple, choose randomly
						return narrow_pred[random.randint(0,len(narrow_pred)-1)]
					elif (len(narrow_pred) == 1):
						return narrow_pred[0]
				elif (len(narrow_pred_next) == 1):
					return narrow_pred_next[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes farther from Predator and predator_next - Condition 2_1 - End

			lcl_dist_to_pred = dist_to_pred.copy()
			lcl_dist_to_pred_next = dist_to_pred_next.copy()
			lcl_dist_to_prey = dist_to_prey.copy()

			#selecting nodes farther from Predators current location - Condition 2_2 - Start
			for i in candidate_nodes:
				if dist_to_pred[i] > min_pred_dist:
					narrowed_candidates.append(i)
				else:
					del lcl_dist_to_pred[i]
					del lcl_dist_to_pred_next[i]
			
			print("Inside condition 2_2 check")

			#Checking if there are multiple nodes that farther from the predator
			if (len(narrowed_candidates) > 1):
				narrow_pred = []
				#Choose nodes that are farthest from predators current position
				narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]

				#Checking if there are multiple nodes that are farther from the predator with the same distance
				if (len(narrow_pred) > 1):
					#remove key, val pairs from distance to predator next position dictionary which are not selected above
					for key in lcl_dist_to_pred.keys():
						if key not in narrow_pred:
							del lcl_dist_to_pred_next[key]

					narrow_pred_next = []
					#Select nodes that are farthest away from the predators next position
					narrow_pred_next = [key for key, value in lcl_dist_to_pred_next.items() if value == max(lcl_dist_to_pred_next.values())]
					if (len(narrow_pred_next) > 1):
							#If multiple, choose randomly
							return narrow_pred_next[random.randint(0,len(narrow_pred_next)-1)]
					elif (len(narrow_pred_next) == 1):
						return narrow_pred_next[0]
				elif (len(narrow_pred) == 1):
					return narrow_pred[0]
			elif(len(narrowed_candidates) == 1):
				return narrowed_candidates[0]
			#selecting nodes farther from Predators current location - Condition 2_2 - End
		return agent_pos


	def decide_node(self, dist_to_prey, dist_to_pred, agent_pos, candidate_nodes):
		min_pred_dist = dist_to_pred[agent_pos]
		min_prey_dist = dist_to_prey[agent_pos]
		chosen_node = agent_pos
		narrowed_candidates = []
		del dist_to_pred[agent_pos]
		del dist_to_prey[agent_pos]
		lcl_dist_to_pred = dist_to_pred.copy()
		lcl_dist_to_prey = dist_to_prey.copy()

		#selecting nodes Closer to Prey and Farther from Predator - Condition 1 - Start
		for i in candidate_nodes:
			#Shortlist all neighbouring nodes that are closer to prey and farther from the predator
			if dist_to_prey[i] < min_prey_dist and dist_to_pred[i] > min_pred_dist:
				narrowed_candidates.append(i)
			else:
				#remove key, val pairs from distance to prey/pred dictionaries for nodes not selected above
				del lcl_dist_to_pred[i]
				del lcl_dist_to_prey[i]
		
		print("Inside condition 1 check")

		#Checking if there are multiple nodes that are closer to prey and farther from the predator
		if (len(narrowed_candidates) > 1):
			narrow_prey = []
			#If multiple, selecting node with minimum distance to prey
			narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]

			#Checking if there are multiple nodes that are closer to prey with the same distance
			if (len(narrow_prey) > 1):
				#remove key, val pairs from distance to predator dictionary which are not selected above
				for key in lcl_dist_to_pred.keys():
					if key not in narrow_prey:
						del lcl_dist_to_pred[key]

				#If multiple, selecting node with maximum distance to predator
				narrow_pred = []
				#If multiple, selecing node with maximum distance from predator
				narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]
				if (len(narrow_pred) > 1):
					#If still multiple after selecting min(distance from prey) and max(distance from predator), selecting randomly
					chosen_node = narrow_pred[random.randint(0,len(narrow_pred)-1)]
					return chosen_node
				elif (len(narrow_pred) == 1):
					return narrow_pred[0]
			elif (len(narrow_prey) == 1):
				return narrow_prey[0]
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes Closer to Prey and Farther from Predator - Condition 1 - End

		lcl_dist_to_prey = dist_to_prey.copy()
		
		#selecting nodes Closer to Prey and not Farther from Predator - Condition 2 - Start
		for i in candidate_nodes:
			if dist_to_prey[i] < min_prey_dist and dist_to_pred[i] == min_pred_dist:
				narrowed_candidates.append(i)
			else:
				del lcl_dist_to_prey[i]
		
		print("Inside condition 2 check")

		if (len(narrowed_candidates) > 1):
			narrow_prey = []
			#If multiple, selecting node with minimum distance to prey
			narrow_prey = [key for key, value in lcl_dist_to_prey.items() if value == min(lcl_dist_to_prey.values())]

			if (len(narrow_prey) > 1):
				chosen_node = narrow_prey[random.randint(0,len(narrow_prey)-1)]
				return chosen_node
			elif (len(narrow_prey) == 1):
				return narrow_prey[0]
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes Closer to Prey and not Farther from Predator - Condition 2 - End

		lcl_dist_to_pred = dist_to_pred.copy()

		#selecting nodes not farther from Prey and Farther from Predator - Condition 3 - Start
		for i in candidate_nodes:
			if dist_to_prey[i] == min_prey_dist and dist_to_pred[i] > min_pred_dist:
				narrowed_candidates.append(i)
			else:
				del lcl_dist_to_pred[i]
		
		print("Inside condition 3 check")

		if (len(narrowed_candidates) > 1):
			narrow_pred = []
			#If multiple, selecting node with minimum distance to prey
			narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]

			if (len(narrow_pred) > 1):
				chosen_node = narrow_pred[random.randint(0,len(narrow_pred)-1)]
				return chosen_node
			elif (len(narrow_pred) == 1):
				return narrow_pred[0]
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes not farther from Prey and Farther from Predator - Condition 3 - End

		#selecting nodes not farther from Prey and not closer to Predator - Condition 4 - Start
		for i in candidate_nodes:
			if dist_to_prey[i] == min_prey_dist and dist_to_pred[i] == min_pred_dist:
				narrowed_candidates.append(i)
		
		print("Inside condition 4 check")

		if (len(narrowed_candidates) > 1):
			chosen_node = narrowed_candidates[random.randint(0,len(narrowed_candidates)-1)]
			return chosen_node
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes not farther from Prey and not closer to Predator - Condition 4 - End

		lcl_dist_to_pred = dist_to_pred.copy()

		#selecting nodes farther from Predator - Condition 5 - Start
		for i in candidate_nodes:
			if dist_to_pred[i] > min_pred_dist:
				narrowed_candidates.append(i)
			else:
				del lcl_dist_to_pred[i]
		
		print("Inside condition 5 check")

		if (len(narrowed_candidates) > 1):
			narrow_pred = []
			#If multiple, selecting node with minimum distance to prey
			narrow_pred = [key for key, value in lcl_dist_to_pred.items() if value == max(lcl_dist_to_pred.values())]

			if (len(narrow_pred) > 1):
				chosen_node = narrow_pred[random.randint(0,len(narrow_pred)-1)]
				return chosen_node
			elif (len(narrow_pred) == 1):
				return narrow_pred[0]
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes farther from Predator - Condition 5 - End

		#selecting nodes not closer to Predator - Condition 6 - Start
		for i in candidate_nodes:
			if dist_to_pred[i] == min_pred_dist:
				narrowed_candidates.append(i)
		
		print("Inside condition 6 check")

		if (len(narrowed_candidates) > 1):
			chosen_node = narrowed_candidates[random.randint(0,len(narrowed_candidates)-1)]
			return chosen_node
		elif(len(narrowed_candidates) == 1):
			return narrowed_candidates[0]
		#selecting nodes not closer to Predator - Condition 6 - End

		#Stay still and Pray - Condition 7 - Start
		print("Inside condition 7")
		return agent_pos
		#Stay still and Pray - Condition 7 - End

	def check_status(self, agent_pos, pred_pos, prey_pos):
		if agent_pos == pred_pos:
			return 2
		if agent_pos == prey_pos:
			return 1
		return 0

	def init_prob(self, graph_dict, pred_pos, agent_pos):
		print("Inside init_prob...")
		if pred_pos == -1:
			print("Initializing prey probabilities...")
			for key in graph_dict.keys():
				if key == agent_pos:
					self.prey_probabilities[key] = 0
				else:
					self.prey_probabilities[key] = 1/(len(graph_dict)-1)
		else:
			print("Initializing pred probabilities...")
			for key in graph_dict.keys():
				if key == pred_pos:
					self.pred_probabilities[key] = 1
				else:
					self.pred_probabilities[key] = 0


	def update_prey_prob_presurvey(self, graph_dict, pred_pos, agent_pos, step_count):
		print("In update_prey_prob_presurvey...")
		prey_probabilities_temp = {}

		if self.last_seen_prey < 0 and step_count == 1:
			print("Prey not yet found...ever...Calculating probability based on that")
			self.init_prob(graph_dict, -1, agent_pos)
		else:
			#P(x) = ∑(P(in x now, was in i)) for i in 0...49 = ∑(P(was in i).P(in x|was in i)) for i in 0...49
			for key in graph_dict.keys():
				temp = 0
				for k in self.prey_probabilities.keys():
					child_prob = 0

					if key in graph_dict[k] or key == k:
						child_prob = (1/(len(graph_dict[k])+1))
					else:
						child_prob = 0
					temp += (self.prey_probabilities[k] * child_prob)
				prey_probabilities_temp[key] = temp

			self.prey_probabilities = prey_probabilities_temp.copy()


	def update_prey_prob_postsurvey(self, graph_dict, prey_pos, agent_pos, step_count, survey_node, survey_result):
		print("In update_prey_prob_postsurvey...")
		prey_post_survey_prob_temp = {}

		if survey_result == True:
			print("Prey found from survey. Calculating probability based on that")
			for key in graph_dict.keys():
				if key == prey_pos:
					prey_post_survey_prob_temp[key] = 1
				else:
					prey_post_survey_prob_temp[key] = 0
		else:
			#P(survey_node) = 0
			#P(x) = P(in X, not in survey_node)/P(not in survey_node)
			#P(x) = P(x).P(not in survey_node|in x) / (∑P(i).P(not in survey_node| in i)) for i in 0...49
			for key in graph_dict.keys():
				if key == survey_node:
					P_not_in_survey_given_in_key = 0
				else:
					P_not_in_survey_given_in_key = 1
				
				P_not_in_survey = 0
				
				for k in self.prey_probabilities.keys():
					if k == survey_node:
						P_denom_second_term = 0
					else:
						P_denom_second_term = 1
					
					P_not_in_survey += (self.prey_probabilities[k]*P_denom_second_term)

				prey_post_survey_prob_temp[key] = self.prey_probabilities[key]*P_not_in_survey_given_in_key/P_not_in_survey

		self.prey_probabilities = prey_post_survey_prob_temp.copy()


	def update_pred_prob_presurvey(self, graph_dict, pred_pos, agent_pos, step_count):
		print("In update_pred_prob_presurvey...")
		pred_probabilities_temp = {}
		graph_prob = Graph()

		if step_count == 1:
			self.init_prob(graph_dict, pred_pos, agent_pos)
		else:
			#∑(P(was in i,random_way,in y) + P(was in i,shortest_way,in y))
			#P(y) = ∑(P(was in i).P(random_way|was in i).P(in y|was in i,random_way) + P(was in i).P(shortest_way|was in i).P(in y|was in i,shortest_way))
			for key in graph_dict.keys():
				temp = 0
				for k in self.pred_probabilities.keys():
					child_prob = 0
					temp_path = []

					if key in graph_dict[k]:# or key == k:
						child_prob = (1/(len(graph_dict[k])))
					else:
						child_prob = 0

					if key in graph_dict[k] and k != agent_pos:
						temp_path = graph_prob.calc_path(graph_dict, k, agent_pos)

						if key == temp_path[1]:
							prob_undictract_next = 1
						else:
							prob_undictract_next = 0
					else:
						prob_undictract_next = 0

					temp += ((self.pred_probabilities[k] * 0.4 * child_prob) + (self.pred_probabilities[k] * 0.6 * prob_undictract_next))
				pred_probabilities_temp[key] = temp

			self.pred_probabilities = pred_probabilities_temp.copy()


	def update_pred_prob_postsurvey(self, graph_dict, pred_pos, agent_pos, step_count, survey_node, survey_result):
		print("In update_pred_prob_postsurvey...")
		pred_post_survey_prob_temp = {}

		if survey_result == True:
			print("Pred found from survey. Calculating probability based on that")
			for key in graph_dict.keys():
				if key == pred_pos:
					pred_post_survey_prob_temp[key] = 1
				else:
					pred_post_survey_prob_temp[key] = 0
		else:
			#P(survey_node) = 0
			#P(x) = P(in X, not in survey_node)/P(not in survey_node)
			#P(x) = P(x).P(not in survey_node|in x) / (∑P(i).P(not in survey_node| in i)) for i in 0...49
			for key in graph_dict.keys():
				if key == survey_node:
					P_not_in_survey_given_in_key = 0
				else:
					P_not_in_survey_given_in_key = 1
				P_not_in_survey = 0
				
				for k in self.pred_probabilities.keys():
					if k == survey_node:
						P_denom_second_term = 0
					else:
						P_denom_second_term = 1

					P_not_in_survey += (self.pred_probabilities[k]*P_denom_second_term)

				if self.pred_probabilities[key]*P_not_in_survey_given_in_key == 0:
					pred_post_survey_prob_temp[key] = 0
				else:
					pred_post_survey_prob_temp[key] = self.pred_probabilities[key]*P_not_in_survey_given_in_key/P_not_in_survey

		self.pred_probabilities = pred_post_survey_prob_temp.copy()

	#Calculates the utility values and keeps on looping until they converge and diff becomes < 0.0001
	def initialize_vals(self, graph_dict):
		done = False
		for agent in range(50):
			for prey in range(50):
				for pred in range(50):
					if agent == prey and prey == pred:
						self.rewards[(agent,prey,pred)] = -1
						self.utility_vals[(agent,prey,pred)] = -1#0
					elif agent == pred:
						#self.rewards[(agent,prey,pred)] = 0
						self.rewards[(agent,prey,pred)] = -1
						self.utility_vals[(agent,prey,pred)] = -1#0
					elif agent == prey:
						self.rewards[(agent,prey,pred)] = 1
						self.utility_vals[(agent,prey,pred)] = 0
					else:
						self.rewards[(agent,prey,pred)] = 0
						self.utility_vals[(agent,prey,pred)] = 0
						if prey in graph_dict[agent]:
							self.starting_pt.append((agent,prey,pred))

		util_vals_cp = copy.deepcopy(self.utility_vals)

		graph_dict_tmp = copy.deepcopy(graph_dict)

		for key in graph_dict_tmp.keys():
			graph_dict_tmp[key].append(key)

		#Calculating path intercept to later check in calc_util_val transition probability of a state. It is needed as neighbors of the predator
		#will have different transition depending on whether the neighboring node to the predator comes in the path from predator to the agent because of distracted predator
		graph_tmp = Graph()
		for pred in range(50):
			for agent in range(50):
				for curr in range(50):
					if (curr in graph_tmp.calc_path(graph_dict, pred, agent)):
						self.path_dict[(pred,agent,curr)] = 1
					else:
						self.path_dict[(pred,agent,curr)] = 0

		#Looping over all possible 125,000 states and calculating the utility value
		ctr = 1
		while(done == False):
			for agent in range(50):
				for prey in range(50):
					for pred in range(50):
						self.calc_util_val(graph_dict, agent, prey, pred, util_vals_cp, graph_dict_tmp)
			check_sum = 0
			for key in self.utility_vals.keys():
				agent_1,prey_1,pred_1 = key
				if (self.utility_vals[(agent_1,prey_1,pred_1)] != util_vals_cp[(agent_1,prey_1,pred_1)]):
					check_sum += (util_vals_cp[(agent_1,prey_1,pred_1)] - self.utility_vals[(agent_1,prey_1,pred_1)])
			print("check_sum: " + str(check_sum))
			#If sum of difference if new and old uitlity values of all states < 0.0001, convergence reached. Can stop and use these utility values
			if check_sum < 0.0001:
				print("Can stop iterating. U_star converged...")
				done = True

			self.utility_vals = copy.deepcopy(util_vals_cp)
			ctr += 1
		return


	#Function that calculates the utility value of the given input state
	def calc_util_val(self, graph_dict, agent_pos, prey_pos, pred_pos, util_vals_cp, graph_dict_tmp, add_flag = False):
		beta = 0.9
		temp_vals = []
		graph_tmp = Graph()

		for agent in graph_dict_tmp[agent_pos]:
			second_term = 0
			for prey in graph_dict_tmp[prey_pos]:
				for pred in graph_dict[pred_pos]:
					#According to the path intercept calcualted in initialize_vals(), assign transition probabilities for distracted predator case
					if (self.path_dict[(pred_pos,agent_pos,pred)] == 1):
						#If neighboring state happens where pred variable node comes in path of predator to agent, we add an extra 0.6 for shortest distance predator case
						temp_val = (0.6 + (0.4 * (1/len(graph_dict[pred_pos]))))
					else:
						#Vanilla distracted predator case where the predator moves randomly
						temp_val = (0.4 * (1/len(graph_dict[pred_pos])))
					#Calculating the transition probability of the input state
					self.transition_prob[(agent,prey,pred)] = (1/len(graph_dict_tmp[prey_pos])) * temp_val
					print("transition_prob (" + str(agent) + "," + str(prey) + "," + str(pred) + "): " + str(self.transition_prob[(agent,prey,pred)]))
					#Multiplying and taking sum of transition probability and utility of possible states for ana ction
					second_term += self.transition_prob[(agent,prey,pred)] * util_vals_cp[(agent,prey,pred)]
			#Calculating utility of the state for an action (runs for all possible actions) and then appending to a list
			second_term = self.rewards[(agent,prey_pos,pred_pos)] + (beta * second_term)
			print("second_term (" + str(agent_pos) + "," + str(prey_pos) + "," + str(pred_pos) + ") to move to " + str(agent) + ": " + str(second_term))
			temp_vals.append([agent,second_term])

		#Selecting the max utility value from the list of utility values for all possible actions
		temp_ustr = temp_vals[0][1]
		for i in range(1,len(temp_vals)):
			if (temp_vals[i][1] > temp_ustr):
				temp_ustr = temp_vals[i][1]

		print("rewards: " + str(self.rewards[(agent_pos,prey_pos,pred_pos)]))
		#Setting utility of input state
		util_vals_cp[(agent_pos,prey_pos,pred_pos)] = temp_ustr
		print("UStar of (" + str(agent_pos) + "," + str(prey_pos) + "," + str(pred_pos) + "): " + str(util_vals_cp[(agent_pos,prey_pos,pred_pos)]))

		#One more iteration of utility value update is done which fetching the utility value of a state for every step in U*
		#Code returns the list of utility values for the current and the possible neighboring states when called from move_agent()[add_flag = True in this case]
		if add_flag == True:
			return temp_vals

		return

	#Function to select the next position for the agent for u star and u partial agents
	def move_agent(self,graph_dict, agent_pos, prey_pos, pred_pos, partial_case = False):
		graph_dict_tmp = copy.deepcopy(graph_dict)
		util_vals_cp = copy.deepcopy(self.utility_vals)

		#Calculation of modified graph dictionary to include the current node in the dictionary values as the agent and the prey can stay in one place
		for key in graph_dict_tmp.keys():
			graph_dict_tmp[key].append(key)
		if (partial_case == False):
			#In case of agent U star, fetch the list of utility values of the current and neighboring states by calling calc_util_val()
			temp_vals = self.calc_util_val(graph_dict, agent_pos, prey_pos, pred_pos, util_vals_cp, graph_dict_tmp, True)
		else:
			#In case of agent U partial, call calc_partial_utility() to get the updated utility values by multiplying the prey probabilities and summing up for a state
			temp_vals = self.calc_partial_utility(graph_dict, agent_pos, prey_pos, pred_pos, util_vals_cp, graph_dict_tmp)

		print("temp_vals")
		print(temp_vals)

		temp_sum = 0
		abs_sum = 0
		temp_ctr = 0

		#Loop required to segregate the below conditions of different possible Utility values
		for i in range(len(graph_dict_tmp[agent_pos])):
			temp_sum += temp_vals[i][1]
			if temp_vals[i][1] < 0:
				temp_ctr += 1
			abs_sum += abs(temp_vals[i][1])

		#if all vals are 0, choose action randomly
		if temp_sum == 0:
			print("All temp_vals 0")
			x = random.randint(0,len(graph_dict_tmp[agent_pos])-1)
			return temp_vals[x][0]

		#If positive values exist, choose action with max positive value
		for i in range(len(graph_dict_tmp[agent_pos])):
			print("Some temp_vals +ve")
			temp_ustr = temp_vals[0][1]
			idx = 0
			for i in range(1,len(temp_vals)):
				if (temp_vals[i][1] > temp_ustr):
					temp_ustr = temp_vals[i][1]
					idx = i
			return temp_vals[idx][0]
	
	#Function to update the position of agent V
	def move_v_agent(self,graph_dict, agent_pos, prey_pos, pred_pos, nn, partial_case = False):
		graph_dict_tmp = copy.deepcopy(graph_dict)
		util_vals_cp = copy.deepcopy(self.utility_vals)
		temp_vals = []

		#Calculation of modified graph dictionary to include the current node in the dictionary values as the agent and the prey can stay in one place
		for key in graph_dict_tmp.keys():
			graph_dict_tmp[key].append(key)
		#For different possible actions, calling the predict() function of the neural network to fetch the utility values of all neighboring states and appending to a list
		for x in graph_dict_tmp[agent_pos]:
			temp = [x, prey_pos, pred_pos]
			input_value = np.array([temp])
			temp_store = nn.predict(input_value)
			temp_vals.append([x, temp_store[0][0]])

		print("temp_vals")
		print(temp_vals)

		temp_sum = 0
		#abs_sum = 0
		cnt_neg = 0
		cnt_pos = 0

		#Loop required to segregate the below conditions of different possible Utility values
		for i in range(len(graph_dict_tmp[agent_pos])):
			temp_sum += temp_vals[i][1]
			if temp_vals[i][1] < 0:
				cnt_neg += 1
			elif temp_vals[i][1] > 0:
				cnt_pos += 1

		#if all vals are 0, choose action randomly
		if temp_sum == 0:
			print("All temp_vals 0")
			x = random.randint(0,len(graph_dict_tmp[agent_pos])-1)
			return temp_vals[x][0]

		#If all non-positive values (some negative and some positive), choose randomly from zero values
		if temp_sum < 0 and cnt_pos == 0 and cnt_neg < len(graph_dict_tmp[agent_pos]):
			temp_arr = []
			idx = 0
			for i in range(len(graph_dict_tmp[agent_pos])):
				if temp_vals[i][1] == 0:
					temp_arr.append(temp_vals[i][1])
			idx = random.randint(0,len(temp_arr)-1)
			return temp_vals[idx][0]

		#If all values are negative, choose least negative value
		#If positive values exist, choose action with max positive value
		for i in range(len(graph_dict_tmp[agent_pos])):
			if (cnt_neg == len(graph_dict_tmp[agent_pos])):
				print("All temp_vals -ve")
			else:
				print("Some temp_vals +ve")
			temp_ustr = temp_vals[0][1]
			idx = 0
			for i in range(1,len(temp_vals)):
				if (temp_vals[i][1] > temp_ustr):
					temp_ustr = temp_vals[i][1]
					idx = i
			return temp_vals[idx][0]

	#Function to calculate the updated utility values for U Partial agent by multiplying and summing the prey probabilities from the belief
	def calc_partial_utility(self, graph_dict, agent_pos, prey_pos, pred_pos, util_vals_cp, graph_dict_tmp):
		print("In calc_partial_utility...")
		temp_vals = []
		for agent in graph_dict_tmp[agent_pos]:
			partial_temp = 0
			print("considering agent move to " + str(agent))
			for prey in range(0,50):
				print("considering for prey position " + str(prey) + " with prey_pos probability " + str(prey_pos[prey]))
				partial_temp += (prey_pos[prey] * util_vals_cp[(agent,prey,pred_pos)])
			temp_vals.append([agent,partial_temp])

		return temp_vals

#Neural Network base layer that is inherited by classes of other types of layers
class Layer:
	def __init__(self):
		self.input = None
		self.output = None

	def forward_propagation(self, input):
		pass

	def backward_propagation(self, output_error, learning_rate):
		pass

#Class for the fully connected layer of the neural network
class FCLayer(Layer):
	def __init__(self, input_size, output_size):
		#Initializing the weights and biases for this layer randomly around mean 0
		self.weights = np.random.rand(input_size, output_size) - 0.5
		self.bias = np.random.rand(1, output_size) - 0.5

	#Function to implement forward propagation for the fully connected layer
	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = np.dot(self.input, self.weights) + self.bias
		return self.output

	#Function to implement backward propagation for the fully connected layer
	def backward_propagation(self, output_error, learning_rate):
		input_error = np.dot(output_error, self.weights.T)
		weights_error = np.dot(self.input.T, output_error)

		self.weights -= learning_rate * weights_error
		self.bias -= learning_rate * output_error
		return input_error

#Class for implementing the Activation layers
class ActivationLayer(Layer):
	#Setting the activation function to leaku ReLU and it's corresponding function taht returns the derivative of leaky ReLU
	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime

	#Function to implement the forward prop of the activation layer by calling the function leaky_relu() and returning the output
	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = self.activation(self.input)
		return self.output

	##Function to implement the backward prop of the activation layer by calling the leaky_relu_back() and returning the derivative output
	def backward_propagation(self, output_error, learning_rate):
		return self.activation_prime(self.input) * output_error

#Class to implement and train the NeuralNet and predict the outputs
class neuralnet():
	input_states = []
	input_states_1 = []
	output_utils = []
	output_utils_1 = []
	def __init__(self):
		self.layers = []
		#self.loss = None
		#self.loss_prime = None
		self.loss = self.calc_error
		self.loss_prime = self.calc_backward_error

	#Function to format the graph(input) and utility values(output) for the neural network
	def read_data(self, graph_v, agents_comm_v):
		for key in agents_comm_v.utility_vals.keys():
			agent,prey,pred = key
			self.input_states.append([[agent,prey,pred]])
			self.output_utils.append([agents_comm_v.utility_vals[key]])


		#Structuring input and output values as numpy arrays
		#np.size, np.resize, np.reshape we necessary to debug the code as we were encoutering issues with the dimensions during the forward and backward prop
		#in matrix multiplication. Using the above numpy functions helped out a lot!
		#the code that was added during debugging is removed now, because there were 100s of lines of prints used for debugging
		self.input_states = np.array(self.input_states)
		print("input_states: " + str(self.input_states))
		print("input_states shape: " + str(self.input_states.shape))

		self.output_utils = np.array(self.output_utils)
		print("output_utils: " + str(self.output_utils))
		print("output_utils.shape: " + str(self.output_utils.shape))
		
		return (self.input_states, self.output_utils)

	#Function used to add the layers to the base layer
	def add(self, layer):
		self.layers.append(layer)

	#Function to predict the utility values when called by the move_v_agent() function
	def predict(self, input_data):
		samples = len(input_data)
		result = []

		#Input length would be 1 in case of predict being called
		#The below code computes one forward pass through the neural network to obtain the uitlity value of the single input state
		for i in range(samples):
			output = input_data[i]
			for layer in self.layers:
				output = layer.forward_propagation(output)
		return output

	#Function to train the neural network on the input of all possible 125,000 states of a graph
	def fit(self, x_train, y_train, learning_rate):
		print("in fit...")
		samples = len(x_train)
		x = 2
		i = 0
		flag = False
		flag_1 = False
		while (x > 0.01):
			err = 0
			#Instead of flattening allinputs into a matrix and performing forward and backward prop on that, this mode
			#runs the loop on a single input state out of the 125,000 states. It then repeats for the entire input size
			for j in range(samples):
				output = x_train[j]
				for layer in self.layers:
					#For all layers(Fully connected and the activation), perform the forward prop
					output = layer.forward_propagation(output)

				#Calculat the error by calling calc_error()
				err += self.loss(y_train[j], output)
				#Compute the derivative of the loss value function and get the value
				error = self.loss_prime(y_train[j], output)

				ctr = 0

				for layer in reversed(self.layers):
					#Perform the backprop for all the layers from the last to the first
					error = layer.backward_propagation(error, learning_rate)

			err /= samples
			x = abs(err)
			i += 1
			#print the error calculated after the forward prop
			print("epoch " + str(i) + " | error: " + str(err))
			#Since we are not using optimization techniques like RMSPros or Adam, had to lower the learning_rate below a loss value through if-else
			#If not done, loss value kept going below zero
			if (err < 0.5):
				learning_rate = 0.000001
			else:
				learning_rate = 0.0001

	#Function to calculate the activation output for the forwad prop
	def leaky_relu(self, x):
		return np.where(x > 0, x, x * 0.01)

	#Function to calculate the activation output for the backward prop
	def leaky_relu_back(self, x):
		#Formula calculated by taking teh derivative of the formula used for forward prop
		return np.where(x > 0, 1, 0.01)
	
	#Function to calculate the mean absolute  loss
	def calc_error(self, y_true, y_pred):
		#return np.mean(abs(y_true - y_pred))
		return np.mean(abs(y_pred - y_true))

	#Derivative of the error function for backprop
	def calc_backward_error(self, y_true, y_pred):
		return np.where(y_true > y_pred, -1, 1)
		

	#Entry point into the neural network to stack the layers and create the neuralnet model for training
	def calculate_utils(self, graph_v, agents_comm_v):
		print("in calculate_utils...")
		#Call to read data to fetch the formatted input and output values
		train_x_data, train_y_data = self.read_data(graph_v, agents_comm_v)

		#Stacking and creating a 6-layer neural network model
		self.layers.append(FCLayer(3, 7))
		self.layers.append(ActivationLayer(self.leaky_relu, self.leaky_relu_back))
		self.layers.append(FCLayer(7, 11))
		self.layers.append(ActivationLayer(self.leaky_relu, self.leaky_relu_back))
		self.layers.append(FCLayer(11, 14))
		self.layers.append(ActivationLayer(self.leaky_relu, self.leaky_relu_back))
		self.layers.append(FCLayer(14, 7))
		self.layers.append(ActivationLayer(self.leaky_relu, self.leaky_relu_back))
		self.layers.append(FCLayer(7, 3))
		self.layers.append(ActivationLayer(self.leaky_relu, self.leaky_relu_back))
		self.layers.append(FCLayer(3, 1))
		self.layers.append(ActivationLayer(self.leaky_relu, self.leaky_relu_back))

		#Calling the fit function to train the model.
		#We run the code until the loss value is close enough to zero instead of running for a fixed number of epochs
		self.fit(train_x_data, train_y_data, learning_rate=0.0001)


#Agent 1 from project 2
#AGENT1_START#
class Agent1():
	max_Step_count = 50
	def proceed(self, graph_dict, connectivity_matrix, prey_pos, pred_pos, agent_pos):
		game_over = 0
		graph_1 = Graph()
		agents_comm_1 = agents_common()
		prey_1 = Prey()
		pred_1 = Predator()

		while (agents_comm_1.step_count <= self.max_Step_count and game_over == 0):
			print("agents_comm_1.step_count: " + str(agents_comm_1.step_count))
			print("agent_pos: " + str(agent_pos))
			print("pred_pos: " + str(pred_pos))
			print("prey_pos: " + str(prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(pred_pos)+") from agent("+str(agent_pos)+")")
			dist_to_pred[agent_pos] = len(graph_1.calc_path(graph_dict, agent_pos, pred_pos))
			print("Calculating distance to prey("+str(prey_pos)+") from agent("+str(agent_pos)+")")
			dist_to_prey[agent_pos] = len(graph_1.calc_path(graph_dict, agent_pos, prey_pos))
			candidate_nodes = []
			
			for val in graph_dict[agent_pos]:
				dist_to_pred[val] = len(graph_1.calc_path(graph_dict, val, pred_pos))
				dist_to_prey[val] = len(graph_1.calc_path(graph_dict, val, prey_pos))
				candidate_nodes.append(val)
			
			print("Agent 1 position before update: "+str(agent_pos))
			agent_pos = agents_comm_1.decide_node(dist_to_prey, dist_to_pred, agent_pos, candidate_nodes)
			print("Agent 1 position after update: "+str(agent_pos))
			game_over = agents_comm_1.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return (game_over,agents_comm_1.step_count)

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_1.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(prey_pos))
			game_over = agents_comm_1.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return (game_over,agents_comm_1.step_count)

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_1.move_pred(graph_dict, agent_pos, pred_pos)
			print("Pred position after update: "+str(pred_pos))
			game_over = agents_comm_1.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return (game_over,agents_comm_1.step_count)
			agents_comm_1.step_count += 1
		return (game_over,agents_comm_1.step_count)
#AGENT1_END#


#Agent 2 from project 2
#AGENT2_START#
class Agent2():
	max_Step_count = 50

	def proceed(self, graph_dict, connectivity_matrix, prey_pos, pred_pos, agent_pos):
		game_over = 0
		graph_2 = Graph()
		agents_comm_2 = agents_common()
		prey_2 = Prey()
		pred_2 = Predator()

		while (agents_comm_2.step_count <= self.max_Step_count and game_over == 0):
			print("agents_comm_2.step_count: " + str(agents_comm_2.step_count))
			print("agent_pos: " + str(agent_pos))
			print("pred_pos: " + str(pred_pos))
			print("prey_pos: " + str(prey_pos))
			possible_locs = [prey_pos]
			dist_to_prey = {}
			dist_to_pred = {}

			for i in graph_dict[prey_pos]:
				possible_locs.append(i)
			print("Calculating distance to pred("+str(pred_pos)+") from agent("+str(agent_pos)+")")
			dist_to_pred[agent_pos] = len(graph_2.calc_path(graph_dict, agent_pos, pred_pos))
			chosen_prey_pos = random.choice(possible_locs)
			if len(graph_2.calc_path(graph_dict, agent_pos, prey_pos)) <= 1:
				chosen_prey_pos = prey_pos
			print("Calculating distance to prey("+str(chosen_prey_pos)+") from agent("+str(agent_pos)+")")
			dist_to_prey[agent_pos] = len(graph_2.calc_path(graph_dict, agent_pos, chosen_prey_pos))
			candidate_nodes = []
			
			for val in graph_dict[agent_pos]:
				dist_to_pred[val] = len(graph_2.calc_path(graph_dict, val, pred_pos))
				dist_to_prey[val] = len(graph_2.calc_path(graph_dict, val, chosen_prey_pos))
				candidate_nodes.append(val)
			
			print("Agent 2 position before update: "+str(agent_pos))
			agent_pos = agents_comm_2.decide_node_even(graph_2.graph_dict, dist_to_prey, dist_to_pred, agent_pos, pred_pos, candidate_nodes)
			print("Agent 2 position after update: "+str(agent_pos))
			game_over = agents_comm_2.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return (game_over,agents_comm_2.step_count)

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_2.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(prey_pos))
			game_over = agents_comm_2.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return (game_over,agents_comm_2.step_count)

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_2.move_pred(graph_dict, agent_pos, pred_pos)
			print("Pred position after update: "+str(pred_pos))
			game_over = agents_comm_2.check_status(agent_pos, pred_pos, prey_pos)
			if game_over != 0:
				return (game_over,agents_comm_2.step_count)
			agents_comm_2.step_count += 1
		return (game_over,agents_comm_2.step_count)
#AGENT2_END#


#Agent 3 from project 2
#AGENT3_START#
class Agent3():
	max_Step_count = 50

	def survey_node(self, graph_dict, prey_pos, prey_possible):
		if prey_pos == prey_possible:
			return True
		return False

	def proceed(self, graph):
		game_over = 0
		agents_comm_3 = agents_common()
		prey_3 = Prey()
		pred_3 = Predator()
		prey_known_ctr = 0

		while (agents_comm_3.step_count <= self.max_Step_count and game_over == 0):
			survey_result = False
			print("agents_comm_3.step_count: " + str(agents_comm_3.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(graph.pred_pos)+") from agent("+str(graph.agent_pos)+")")
			dist_to_pred[graph.agent_pos] = len(graph.calc_path(graph.graph_dict, graph.agent_pos, graph.pred_pos))

			agents_comm_3.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_3.step_count)
			print("Prey Probabilities before survey: " + str(agents_comm_3.prey_probabilities))
			print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_3.prey_probabilities.values())))

			max_prey_prob_nodes = []
			max_prey_prob_nodes = [key for key, value in agents_comm_3.prey_probabilities.items() if value == max(agents_comm_3.prey_probabilities.values())]
			print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
			if (len(max_prey_prob_nodes) == 1):
				prey_possible = max_prey_prob_nodes[0]
			else:
				prey_possible = random.choice(max_prey_prob_nodes)
			
			print("Surveying Node: " + str(prey_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.prey_pos, prey_possible)

			agents_comm_3.update_prey_prob_postsurvey(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_3.step_count, prey_possible, survey_result)
			print("Updated Prey probabilities after survey: " + str(agents_comm_3.prey_probabilities))
			print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_3.prey_probabilities.values())))

			if survey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_3.last_seen_prey = agents_comm_3.step_count
				prey_known_ctr += 1
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_3.prey_probabilities.items() if value == max(agents_comm_3.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			print("Calculating distance to prey("+str(lcl_prey_pos)+") from agent("+str(graph.agent_pos)+")")
			dist_to_prey[graph.agent_pos] = len(graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_prey_pos))

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				dist_to_pred[val] = len(graph.calc_path(graph.graph_dict, val, graph.pred_pos))
				dist_to_prey[val] = len(graph.calc_path(graph.graph_dict, val, lcl_prey_pos))
				candidate_nodes.append(val)

			print("Agent 3 position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_3.decide_node(dist_to_prey, dist_to_pred, graph.agent_pos, candidate_nodes)
			print("Agent 3 position after update: "+str(graph.agent_pos))
			game_over = agents_comm_3.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_3.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return (game_over,agents_comm_3.step_count)

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_3.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_3.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_3.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return (game_over,agents_comm_3.step_count)

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_3.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_3.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_3.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return (game_over,agents_comm_3.step_count)

			agents_comm_3.step_count += 1

		if agents_comm_3.step_count > self.max_Step_count:
			prey_known_percent = (prey_known_ctr/(agents_comm_3.step_count-1))*100
		else:
			prey_known_percent = (prey_known_ctr/agents_comm_3.step_count)*100
			
		print("Prey Known Percentage: " + str(prey_known_percent))

		return (game_over,agents_comm_3.step_count)
#AGENT3_END#


#Agent 4 from project 2
#AGENT4_START#
class Agent4():
	max_Step_count = 50

	def survey_node(self, graph_dict, prey_pos, prey_possible):
		if prey_pos == prey_possible:
			return True
		return False

	def proceed(self, graph):
		game_over = 0
		agents_comm_4 = agents_common()
		prey_4 = Prey()
		pred_4 = Predator()
		pre_prob_backup = {}
		prey_next_calculated = False
		prey_known_ctr = 0

		while (agents_comm_4.step_count <= self.max_Step_count and game_over == 0):
			survey_result = False
			print("agents_comm_4.step_count: " + str(agents_comm_4.step_count))
			print("graph.agent_pos: " + str(graph.agent_pos))
			print("graph.pred_pos: " + str(graph.pred_pos))
			print("graph.prey_pos: " + str(graph.prey_pos))
			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(graph.pred_pos)+") from agent("+str(graph.agent_pos)+")")
			dist_to_pred[graph.agent_pos] = len(graph.calc_path(graph.graph_dict, graph.agent_pos, graph.pred_pos))

			if (prey_next_calculated == False):
				agents_comm_4.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_4.step_count)
				print("Prey Probabilities before survey: " + str(agents_comm_4.prey_probabilities))
				print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_4.prey_probabilities.values())))
			else:
				prey_next_calculated = False

			max_prey_prob_nodes = []
			max_prey_prob_nodes = [key for key, value in agents_comm_4.prey_probabilities.items() if value == max(agents_comm_4.prey_probabilities.values())]
			print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
			if (len(max_prey_prob_nodes) == 1):
				prey_possible = max_prey_prob_nodes[0]
			else:
				prey_possible = random.choice(max_prey_prob_nodes)
			
			print("Surveying Node: " + str(prey_possible))
			survey_result = self.survey_node(graph.graph_dict, graph.prey_pos, prey_possible)

			agents_comm_4.update_prey_prob_postsurvey(graph.graph_dict, graph.prey_pos, graph.agent_pos, agents_comm_4.step_count, prey_possible, survey_result)
			print("Updated Prey probabilities after survey: " + str(agents_comm_4.prey_probabilities))
			print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_4.prey_probabilities.values())))

			if survey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_4.last_seen_prey = agents_comm_4.step_count
				prey_known_ctr += 1
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_4.prey_probabilities.items() if value == max(agents_comm_4.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			if len(graph.calc_path(graph.graph_dict, graph.agent_pos, lcl_prey_pos)) <= 2:
				chosen_prey_pos = lcl_prey_pos
			else:
				pre_prob_backup = agents_comm_4.prey_probabilities.copy()
				agents_comm_4.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_4.step_count)
				prey_next_calculated = True				
				max_prey_prob_nodes = []
				max_prey_prob_nodes = [key for key, value in agents_comm_4.prey_probabilities.items() if value == max(agents_comm_4.prey_probabilities.values())]
				print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
				if (len(max_prey_prob_nodes) == 1):
					chosen_prey_pos = max_prey_prob_nodes[0]
				else:
					chosen_prey_pos = random.choice(max_prey_prob_nodes)

			print("Calculating distance to prey("+str(chosen_prey_pos)+") from agent("+str(graph.agent_pos)+")")
			dist_to_prey[graph.agent_pos] = len(graph.calc_path(graph.graph_dict, graph.agent_pos, chosen_prey_pos))

			candidate_nodes = []

			for val in graph.graph_dict[graph.agent_pos]:
				dist_to_pred[val] = len(graph.calc_path(graph.graph_dict, val, graph.pred_pos))
				dist_to_prey[val] = len(graph.calc_path(graph.graph_dict, val, chosen_prey_pos))
				candidate_nodes.append(val)

			print("Agent 3 position before update: "+str(graph.agent_pos))
			graph.agent_pos = agents_comm_4.decide_node_even(graph.graph_dict, dist_to_prey, dist_to_pred, graph.agent_pos, graph.pred_pos, candidate_nodes)
			print("Agent 3 position after update: "+str(graph.agent_pos))
			game_over = agents_comm_4.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_4.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return (game_over,agents_comm_4.step_count)

			print("Prey position before update: "+str(graph.prey_pos))
			graph.prey_pos = prey_4.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(graph.prey_pos))
			game_over = agents_comm_4.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_4.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return (game_over,agents_comm_4.step_count)

			print("Pred position before update: "+str(graph.pred_pos))
			graph.pred_pos = pred_4.move_pred(graph.graph_dict, graph.agent_pos, graph.pred_pos)
			print("Pred position after update: "+str(graph.pred_pos))
			game_over = agents_comm_4.check_status(graph.agent_pos, graph.pred_pos, graph.prey_pos)
			if game_over != 0:
				prey_known_percent = (prey_known_ctr/agents_comm_4.step_count)*100
				print("Prey Known Percentage: " + str(prey_known_percent))
				return (game_over,agents_comm_4.step_count)

			agents_comm_4.step_count += 1

		if agents_comm_4.step_count > self.max_Step_count:
			prey_known_percent = (prey_known_ctr/(agents_comm_4.step_count-1))*100
		else:
			prey_known_percent = (prey_known_ctr/agents_comm_4.step_count)*100
			
		print("Prey Known Percentage: " + str(prey_known_percent))

		return (game_over,agents_comm_4.step_count)
#AGENT4_END#


#AGENT U star_START#
class Agent_u_star():
	max_Step_count = 300
	#Function runs the game until any game-ending condition is met. If met, it returns the outcome(win, lose, disbanded) and the latest step count
	def proceed(self, graph_dict, prey_pos, pred_pos, agent_pos, agents_comm_ustr):
		game_over = 0
		agents_comm_ustr.step_count = 1
		prey_ustr = Prey()
		pred_ustr = Predator()

		while (agents_comm_ustr.step_count <= self.max_Step_count and game_over == 0):
			print("agents_comm_ustr.step_count: " + str(agents_comm_ustr.step_count))
			print("agent_pos: " + str(agent_pos))
			print("prey_pos: " + str(prey_pos))
			print("pred_pos: " + str(pred_pos))

			print("Agent position before update: "+str(agent_pos))
			#Call to move_agent() function to fetch the utilty of all the possible states from the current state and updating the agent position
			agent_pos = agents_comm_ustr.move_agent(graph_dict, agent_pos, prey_pos, pred_pos)
			print("Agent position after update: "+str(agent_pos))
			#Check if any game-ending state has occured
			game_over = agents_comm_ustr.check_status(agent_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if game_over != 0:
				return (game_over,agents_comm_ustr.step_count)

			print("Prey position before update: "+str(prey_pos))
			#Function call to move the prey randomly
			prey_pos = prey_ustr.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(prey_pos))
			#Check if any game-ending state has occured
			game_over = agents_comm_ustr.check_status(agent_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if game_over != 0:
				return (game_over,agents_comm_ustr.step_count)

			print("Pred position before update: "+str(pred_pos))
			#Function call to move the distracted predator
			pred_pos = pred_ustr.move_pred(graph_dict, agent_pos, pred_pos)
			print("Pred position after update: "+str(pred_pos))
			#Check if any game-ending state has occured
			game_over = agents_comm_ustr.check_status(agent_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if game_over != 0:
				return (game_over,agents_comm_ustr.step_count)
			agents_comm_ustr.step_count += 1

		return (game_over,agents_comm_ustr.step_count)
#AGENT U star_END#

#AGENT U partial_START#
class Agent_u_partial():
	max_Step_count = 300
	survey_allowed = True
	def survey_node(self, graph_dict, prey_pos, prey_possible):
		if prey_pos == prey_possible:
			return True
		return False

	#def proceed(self, graph, agents_comm_upar):
	def proceed(self, graph_dict, prey_pos, pred_pos, agent_pos, agents_comm_upar):
		game_over = 0
		prey_upar = Prey()
		pred_upar = Predator()
		prey_known_ctr = 0
		agents_comm_upar.step_count = 1

		while (agents_comm_upar.step_count <= self.max_Step_count and game_over == 0):# and ctr == 0):
			print("agents_comm_upar.step_count: " + str(agents_comm_upar.step_count))
			print("agent_pos: " + str(agent_pos))
			print("prey_pos: " + str(prey_pos))
			print("pred_pos: " + str(pred_pos))

			#agents_comm_upar.update_prey_prob_presurvey(graph.graph_dict, graph.pred_pos, graph.agent_pos, agents_comm_upar.step_count)
			agents_comm_upar.update_prey_prob_presurvey(graph_dict, pred_pos, agent_pos, agents_comm_upar.step_count)
			print("Prey Probabilities before survey: " + str(agents_comm_upar.prey_probabilities))
			print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_upar.prey_probabilities.values())))

			#Comment code if want to test U partial without surveying - Start
			if survey_allowed == True:
				max_prey_prob_nodes = []
				max_prey_prob_nodes = [key for key, value in agents_comm_upar.prey_probabilities.items() if value == max(agents_comm_upar.prey_probabilities.values())]
				print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
				if (len(max_prey_prob_nodes) == 1):
					prey_possible = max_prey_prob_nodes[0]
				else:
					prey_possible = random.choice(max_prey_prob_nodes)
			
				print("Surveying Node: " + str(prey_possible))
				#survey_result = self.survey_node(graph.graph_dict, graph.prey_pos, prey_possible)
				survey_result = self.survey_node(graph_dict, prey_pos, prey_possible)

				agents_comm_upar.update_prey_prob_postsurvey(graph_dict, prey_pos, agent_pos, agents_comm_upar.step_count, prey_possible, survey_result)
				print("Updated Prey probabilities after survey: " + str(agents_comm_upar.prey_probabilities))
				print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_upar.prey_probabilities.values())))
			#Comment code if want to test U partial without surveying - End

			print("Agent position before update: "+str(agent_pos))
			agent_pos = agents_comm_upar.move_agent(graph_dict, agent_pos, agents_comm_upar.prey_probabilities, pred_pos, True)
			print("Agent position after update: "+str(agent_pos))
			#Check if any game-ending state has occured
			game_over = agents_comm_upar.check_status(agent_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if game_over != 0:
				return (game_over,agents_comm_upar.step_count)

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_upar.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(prey_pos))
			#Check if any game-ending state has occured
			game_over = agents_comm_upar.check_status(agent_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if game_over != 0:
				return (game_over,agents_comm_upar.step_count)

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_upar.move_pred(graph_dict, agent_pos, pred_pos)
			print("Pred position after update: "+str(pred_pos))
			#Check if any game-ending state has occured
			game_over = agents_comm_upar.check_status(agent_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if game_over != 0:
				return (game_over,agents_comm_upar.step_count)
			agents_comm_upar.step_count += 1
			#ctr += 1
		return (game_over,agents_comm_upar.step_count)
#AGENT U partial_END#


#AGENT V_START#
class Agent_v():
	max_Step_count = 300
	def proceed(self, graph_dict, prey_pos, pred_pos, agent_pos, agents_comm_v, nn):
		game_over = 0
		agents_comm_v.step_count = 1
		#graph_v = Graph()
		prey_v = Prey()
		pred_v = Predator()
		while (agents_comm_v.step_count <= self.max_Step_count and game_over == 0):# and ctr == 0):
			print("agents_comm_v.step_count: " + str(agents_comm_v.step_count))
			print("agent_pos: " + str(agent_pos))
			print("prey_pos: " + str(prey_pos))
			print("pred_pos: " + str(pred_pos))

			print("Agent position before update: "+str(agent_pos))
			agent_pos = agents_comm_v.move_v_agent(graph_dict, agent_pos, prey_pos, pred_pos, nn, False)
			print("Agent position after update: "+str(agent_pos))
			#Check if any game-ending state has occured
			game_over = agents_comm_v.check_status(agent_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if game_over != 0:
				return (game_over,agents_comm_v.step_count)

			print("Prey position before update: "+str(prey_pos))
			#Call to the predict() function of the neural net to move the agent
			prey_pos = prey_v.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(prey_pos))
			#Check if any game-ending state has occured
			game_over = agents_comm_v.check_status(agent_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if game_over != 0:
				return (game_over,agents_comm_v.step_count)

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_v.move_pred(graph_dict, agent_pos, pred_pos)
			print("Pred position after update: "+str(pred_pos))
			#Check if any game-ending state has occured
			game_over = agents_comm_v.check_status(agent_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if game_over != 0:
				#KAUTILYAATULJOSHIreturn game_over
				return (game_over,agents_comm_v.step_count)
			agents_comm_v.step_count += 1
			#ctr += 1
		return (game_over,agents_comm_v.step_count)
#AGENT V_END#

#Running agent U star and agent 1 on the same map at the same time for comparison (combination of both the agent classes)
#Note: the predator in the shortest distance case(subset case of distracted predator), moves towards the agent
#if we chose each agent randomly before every move the predator takes, it may keep on switching from one agent to the other
#This in turn will worsen the predator's performance, by a lot...
#So, so make the case more difficult for out utility based agents, the predator, if not moves randomly, moves towards the new utility agent
#If U star and agent 1 are in the same node, good for the predator. If not, good for agent 1
class Agent_u_star_n_1():
	max_Step_count = 300
	def proceed(self, graph_dict, prey_pos, pred_pos, agent_pos, agents_comm_ustr_n_1, agent_1):
		game_over = 0
		agents_comm_ustr_n_1.step_count = 1
		prey_ustr = Prey()
		pred_ustr = Predator()
		graph_1 = Graph()
		agent_ustr_pos = agent_pos
		agent_1_pos = agent_pos

		while (agents_comm_ustr_n_1.step_count <= self.max_Step_count and game_over == 0):# and ctr == 0):
			print("agents_comm_ustr_n_1.step_count: " + str(agents_comm_ustr_n_1.step_count))
			print("agent_ustr_pos: " + str(agent_ustr_pos))
			print("agent_1_pos: " + str(agent_1_pos))
			print("prey_pos: " + str(prey_pos))
			print("pred_pos: " + str(pred_pos))

			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(pred_pos)+") from agent("+str(agent_1_pos)+")")
			dist_to_pred[agent_1_pos] = len(graph_1.calc_path(graph_dict, agent_1_pos, pred_pos))
			print("Calculating distance to prey("+str(prey_pos)+") from agent("+str(agent_1_pos)+")")
			dist_to_prey[agent_1_pos] = len(graph_1.calc_path(graph_dict, agent_1_pos, prey_pos))
			candidate_nodes = []
			
			for val in graph_dict[agent_1_pos]:
				dist_to_pred[val] = len(graph_1.calc_path(graph_dict, val, pred_pos))
				dist_to_prey[val] = len(graph_1.calc_path(graph_dict, val, prey_pos))
				candidate_nodes.append(val)
			
			print("Agent 1 position before update: "+str(agent_1_pos))
			agent_1_pos = agents_comm_ustr_n_1.decide_node(dist_to_prey, dist_to_pred, agent_1_pos, candidate_nodes)
			print("Agent 1 position after update: "+str(agent_1_pos))

			print("Agent ustr position before update: "+str(agent_ustr_pos))
			agent_ustr_pos = agents_comm_ustr_n_1.move_agent(graph_dict, agent_ustr_pos, prey_pos, pred_pos)
			print("Agent ustr position after update: "+str(agent_ustr_pos))
			#Check if any game-ending state has occured for either of the agents
			game_over_1 = agents_comm_ustr_n_1.check_status(agent_ustr_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_ustr_n_1.check_status(agent_1_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_ustr_pos))
				print("agent_1_pos: " + str(agent_1_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_ustr_n_1.step_count)
				else:
					return (game_over_2,agents_comm_ustr_n_1.step_count)

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_ustr.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(prey_pos))
			#Check if any game-ending state has occured for either of the agents
			game_over_1 = agents_comm_ustr_n_1.check_status(agent_ustr_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_ustr_n_1.check_status(agent_1_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_ustr_pos))
				print("agent_1_pos: " + str(agent_1_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_ustr_n_1.step_count)
				else:
					return (game_over_2,agents_comm_ustr_n_1.step_count)

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_ustr.move_pred(graph_dict, agent_ustr_pos, pred_pos)
			print("Pred position after update: "+str(pred_pos))
			#Check if any game-ending state has occured for either of the agents
			game_over_1 = agents_comm_ustr_n_1.check_status(agent_ustr_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_ustr_n_1.check_status(agent_1_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_ustr_pos))
				print("agent_1_pos: " + str(agent_1_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_ustr_n_1.step_count)
				else:
					return (game_over_2,agents_comm_ustr_n_1.step_count)
			agents_comm_ustr_n_1.step_count += 1
		return (game_over,agents_comm_ustr_n_1.step_count)


#Running agent U star and agent 2 on the same map at the same time for comparison (combination of both the agent classes)
class Agent_u_star_n_2():
	max_Step_count = 300
	def proceed(self, graph_dict, prey_pos, pred_pos, agent_pos, agents_comm_ustr_n_2, agent_2):
		game_over = 0
		agents_comm_ustr_n_2.step_count = 1
		prey_ustr = Prey()
		pred_ustr = Predator()
		graph_2 = Graph()
		agent_ustr_pos = agent_pos
		agent_2_pos = agent_pos

		while (agents_comm_ustr_n_2.step_count <= self.max_Step_count and game_over == 0):# and ctr == 0):
			print("agents_comm_ustr_n_2.step_count: " + str(agents_comm_ustr_n_2.step_count))
			print("agent_ustr_pos: " + str(agent_ustr_pos))
			print("agent_2_pos: " + str(agent_2_pos))
			print("prey_pos: " + str(prey_pos))
			print("pred_pos: " + str(pred_pos))
			
			possible_locs = [prey_pos]
			dist_to_prey = {}
			dist_to_pred = {}
			
			for i in graph_dict[prey_pos]:
				possible_locs.append(i)

			print("Calculating distance to pred("+str(pred_pos)+") from agent("+str(agent_2_pos)+")")
			dist_to_pred[agent_2_pos] = len(graph_2.calc_path(graph_dict, agent_2_pos, pred_pos))
			chosen_prey_pos = random.choice(possible_locs)
			if len(graph_2.calc_path(graph_dict, agent_2_pos, prey_pos)) <= 1:
				chosen_prey_pos = prey_pos
			print("Calculating distance to prey("+str(chosen_prey_pos)+") from agent("+str(agent_2_pos)+")")
			dist_to_prey[agent_2_pos] = len(graph_2.calc_path(graph_dict, agent_2_pos, chosen_prey_pos))
			candidate_nodes = []
			
			for val in graph_dict[agent_2_pos]:
				dist_to_pred[val] = len(graph_2.calc_path(graph_dict, val, pred_pos))
				dist_to_prey[val] = len(graph_2.calc_path(graph_dict, val, chosen_prey_pos))
				candidate_nodes.append(val)
			
			print("Agent 2 position before update: "+str(agent_2_pos))
			agent_2_pos = agents_comm_2.decide_node_even(graph_2.graph_dict, dist_to_prey, dist_to_pred, agent_2_pos, pred_pos, candidate_nodes)
			print("Agent 2 position after update: "+str(agent_2_pos))
			game_over = agents_comm_ustr_n_2.check_status(agent_2_pos, pred_pos, prey_pos)

			print("Agent ustr position before update: "+str(agent_ustr_pos))
			agent_ustr_pos = agents_comm_ustr_n_2.move_agent(graph_dict, agent_ustr_pos, prey_pos, pred_pos)
			print("Agent ustr position after update: "+str(agent_ustr_pos))
			#Check if any game-ending state has occured for either of the agents
			game_over_1 = agents_comm_ustr_n_2.check_status(agent_ustr_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_ustr_n_2.check_status(agent_2_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_ustr_pos))
				print("agent_2_pos: " + str(agent_2_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_ustr_n_2.step_count)
				else:
					return (game_over_2,agents_comm_ustr_n_2.step_count)

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_ustr.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(prey_pos))
			#Check if any game-ending state has occured for either of the agents
			game_over_1 = agents_comm_ustr_n_2.check_status(agent_ustr_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_ustr_n_2.check_status(agent_2_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_ustr_pos))
				print("agent_2_pos: " + str(agent_2_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_ustr_n_2.step_count)
				else:
					return (game_over_2,agents_comm_ustr_n_2.step_count)

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_ustr.move_pred(graph_dict, agent_ustr_pos, pred_pos)
			print("Pred position after update: "+str(pred_pos))
			#Check if any game-ending state has occured for either of the agents
			game_over_1 = agents_comm_ustr_n_2.check_status(agent_ustr_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_ustr_n_2.check_status(agent_2_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_ustr_pos))
				print("agent_2_pos: " + str(agent_2_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_ustr_n_2.step_count)
				else:
					return (game_over_2,agents_comm_ustr_n_2.step_count)
			agents_comm_ustr_n_2.step_count += 1
		return (game_over,agents_comm_ustr_n_2.step_count)


#Class to run agent U partial and agent 3 (Combination of both the agent classes)
class Agent_u_partial_n_3():
	max_Step_count = 300
	survey_allowed = True
	def survey_node(self, graph_dict, prey_pos, prey_possible):
		if prey_pos == prey_possible:
			return True
		return False

	def proceed(self, graph_dict, prey_pos, pred_pos, agent_pos, agents_comm_upar_n_3):
		game_over = 0
		prey_upar = Prey()
		pred_upar = Predator()
		prey_known_ctr = 0
		agents_comm_upar_n_3.step_count = 1
		agent_upar_pos = agent_pos
		agent_3_pos = agent_pos
		graph_3 = Graph()

		while (agents_comm_upar_n_3.step_count <= self.max_Step_count and game_over == 0):# and ctr == 0):
			print("agents_comm_upar_n_3.step_count: " + str(agents_comm_upar_n_3.step_count))
			print("agent_upar_pos: " + str(agent_upar_pos))
			print("agent_3_pos: " + str(agent_3_pos))
			print("prey_pos: " + str(prey_pos))
			print("pred_pos: " + str(pred_pos))

			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(pred_pos)+") from agent("+str(agent_3_pos)+")")
			dist_to_pred[agent_3_pos] = len(graph_3.calc_path(graph_dict, agent_3_pos, pred_pos))

			agents_comm_upar_n_3.update_prey_prob_presurvey(graph_dict, pred_pos, agent_upar_pos, agents_comm_upar_n_3.step_count)
			print("Prey Probabilities before survey: " + str(agents_comm_upar_n_3.prey_probabilities))
			print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_upar_n_3.prey_probabilities.values())))

			#Comment code if want to test U partial without surveying - Start
			
			max_prey_prob_nodes = []
			max_prey_prob_nodes = [key for key, value in agents_comm_upar_n_3.prey_probabilities.items() if value == max(agents_comm_upar_n_3.prey_probabilities.values())]
			print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
			if (len(max_prey_prob_nodes) == 1):
				prey_possible = max_prey_prob_nodes[0]
			else:
				prey_possible = random.choice(max_prey_prob_nodes)
		
			print("Surveying Node: " + str(prey_possible))
			survey_result = self.survey_node(graph_dict, prey_pos, prey_possible)

			agents_comm_upar_n_3.update_prey_prob_postsurvey(graph_dict, prey_pos, agent_upar_pos, agents_comm_upar_n_3.step_count, prey_possible, survey_result)
			print("Updated Prey probabilities after survey: " + str(agents_comm_upar_n_3.prey_probabilities))
			print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_upar_n_3.prey_probabilities.values())))

			if survey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_upar_n_3.last_seen_prey = agents_comm_upar_n_3.step_count
				prey_known_ctr += 1
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_upar_n_3.prey_probabilities.items() if value == max(agents_comm_upar_n_3.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			print("Calculating distance to prey("+str(lcl_prey_pos)+") from agent("+str(agent_3_pos)+")")
			dist_to_prey[agent_3_pos] = len(graph_3.calc_path(graph_dict, agent_3_pos, lcl_prey_pos))

			candidate_nodes = []

			for val in graph_dict[agent_3_pos]:
				dist_to_pred[val] = len(graph_3.calc_path(graph_dict, val, pred_pos))
				dist_to_prey[val] = len(graph_3.calc_path(graph_dict, val, lcl_prey_pos))
				candidate_nodes.append(val)

			print("Agent 3 position before update: "+str(agent_3_pos))
			agent_3_pos = agents_comm_upar_n_3.decide_node(dist_to_prey, dist_to_pred, agent_3_pos, candidate_nodes)
			print("Agent 3 position after update: "+str(agent_3_pos))
			#Comment code if want to test U partial without surveying - End

			print("Agent position before update: "+str(agent_upar_pos))
			agent_upar_pos = agents_comm_upar_n_3.move_agent(graph_dict, agent_upar_pos, agents_comm_upar_n_3.prey_probabilities, pred_pos, True)
			print("Agent position after update: "+str(agent_upar_pos))
			#Check if any game-ending state has occured
			game_over_1 = agents_comm_upar_n_3.check_status(agent_upar_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_upar_n_3.check_status(agent_3_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_upar_pos))
				print("agent_3_pos: " + str(agent_3_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_upar_n_3.step_count)
				else:
					return (game_over_2,agents_comm_upar_n_3.step_count)

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_upar.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(prey_pos))
			#Check if any game-ending state has occured
			game_over_1 = agents_comm_upar_n_3.check_status(agent_upar_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_upar_n_3.check_status(agent_3_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_upar_pos))
				print("agent_3_pos: " + str(agent_3_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_upar_n_3.step_count)
				else:
					return (game_over_2,agents_comm_upar_n_3.step_count)

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_upar.move_pred(graph_dict, agent_upar_pos, pred_pos)
			print("Pred position after update: "+str(pred_pos))
			#Check if any game-ending state has occured
			game_over_1 = agents_comm_upar_n_3.check_status(agent_upar_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_upar_n_3.check_status(agent_3_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_upar_pos))
				print("agent_3_pos: " + str(agent_3_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_upar_n_3.step_count)
				else:
					return (game_over_2,agents_comm_upar_n_3.step_count)
			agents_comm_upar_n_3.step_count += 1
		return (game_over,agents_comm_upar_n_3.step_count)


#Class to run agent U partial and agent 4 (Combination of both the agent classes)
class Agent_u_partial_n_4():
	max_Step_count = 300
	survey_allowed = True
	def survey_node(self, graph_dict, prey_pos, prey_possible):
		if prey_pos == prey_possible:
			return True
		return False

	def proceed(self, graph_dict, prey_pos, pred_pos, agent_pos, agents_comm_upar_n_4):
		game_over = 0
		prey_upar = Prey()
		pred_upar = Predator()
		prey_known_ctr = 0
		agents_comm_upar_n_4.step_count = 1
		agent_upar_pos = agent_pos
		agent_4_pos = agent_pos
		graph_4 = Graph()
		while (agents_comm_upar_n_4.step_count <= self.max_Step_count and game_over == 0):# and ctr == 0):
			print("agents_comm_upar_n_4.step_count: " + str(agents_comm_upar_n_4.step_count))
			print("agent_upar_pos: " + str(agent_upar_pos))
			print("agent_4_pos: " + str(agent_4_pos))
			print("prey_pos: " + str(prey_pos))
			print("pred_pos: " + str(pred_pos))

			dist_to_prey = {}
			dist_to_pred = {}
			print("Calculating distance to pred("+str(pred_pos)+") from agent("+str(agent_4_pos)+")")
			dist_to_pred[agent_4_pos] = len(graph_4.calc_path(graph_dict, agent_4_pos, pred_pos))

			agents_comm_upar_n_4.update_prey_prob_presurvey(graph_dict, pred_pos, agent_pos, agents_comm_upar_n_4.step_count)
			print("Prey Probabilities before survey: " + str(agents_comm_upar_n_4.prey_probabilities))
			print("Sum of Prey Probabilities before survey: " + str(sum(agents_comm_upar_n_4.prey_probabilities.values())))

			#Comment code if want to test U partial without surveying - Start
			if self.survey_allowed == True:
				max_prey_prob_nodes = []
				max_prey_prob_nodes = [key for key, value in agents_comm_upar_n_4.prey_probabilities.items() if value == max(agents_comm_upar_n_4.prey_probabilities.values())]
				print("Getting nodes with max prey probability: " + str(max_prey_prob_nodes))
				if (len(max_prey_prob_nodes) == 1):
					prey_possible = max_prey_prob_nodes[0]
				else:
					prey_possible = random.choice(max_prey_prob_nodes)
			
				print("Surveying Node: " + str(prey_possible))
				#survey_result = self.survey_node(graph.graph_dict, graph.prey_pos, prey_possible)
				survey_result = self.survey_node(graph_dict, prey_pos, prey_possible)

				agents_comm_upar_n_4.update_prey_prob_postsurvey(graph_dict, prey_pos, agent_pos, agents_comm_upar_n_4.step_count, prey_possible, survey_result)
				print("Updated Prey probabilities after survey: " + str(agents_comm_upar_n_4.prey_probabilities))
				print("Sum of updated Prey probabilities after survey: " + str(sum(agents_comm_upar_n_4.prey_probabilities.values())))
			if survey_result == True:
				print("Prey at survey Node!")
				lcl_prey_pos = prey_possible
				agents_comm_upar_n_4.last_seen_prey = agents_comm_upar_n_4.step_count
				prey_known_ctr += 1
			else:
				print("Prey NOT at survey Node!")
				max_prey_prob_nodes = [key for key, value in agents_comm_upar_n_4.prey_probabilities.items() if value == max(agents_comm_upar_n_4.prey_probabilities.values())]
				if (len(max_prey_prob_nodes) == 1):
					lcl_prey_pos = max_prey_prob_nodes[0]
				else:
					lcl_prey_pos = random.choice(max_prey_prob_nodes)

			print("Calculating distance to prey("+str(lcl_prey_pos)+") from agent("+str(agent_4_pos)+")")
			dist_to_prey[agent_4_pos] = len(graph_4.calc_path(graph_dict, agent_4_pos, lcl_prey_pos))

			candidate_nodes = []

			for val in graph_dict[agent_4_pos]:
				dist_to_pred[val] = len(graph_4.calc_path(graph_dict, val, pred_pos))
				dist_to_prey[val] = len(graph_4.calc_path(graph_dict, val, lcl_prey_pos))
				candidate_nodes.append(val)

			print("Agent 4 position before update: "+str(agent_4_pos))
			agent_4_pos = agents_comm_upar_n_4.decide_node(dist_to_prey, dist_to_pred, agent_4_pos, candidate_nodes)
			print("Agent 4 position after update: "+str(agent_4_pos))

			#Comment code if want to test U partial without surveying - End

			print("Agent position before update: "+str(agent_pos))
			agent_pos = agents_comm_upar_n_4.move_agent(graph_dict, agent_pos, agents_comm_upar_n_4.prey_probabilities, pred_pos, True)
			print("Agent position after update: "+str(agent_pos))
			#Check if any game-ending state has occured
			game_over_1 = agents_comm_upar_n_4.check_status(agent_upar_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_upar_n_4.check_status(agent_4_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_upar_pos))
				print("agent_4_pos: " + str(agent_4_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_upar_n_4.step_count)
				else:
					return (game_over_2,agents_comm_upar_n_4.step_count)

			print("Prey position before update: "+str(prey_pos))
			prey_pos = prey_upar.move_prey(graph_dict, prey_pos)
			print("Prey position after update: "+str(prey_pos))
			#Check if any game-ending state has occured
			game_over_1 = agents_comm_upar_n_4.check_status(agent_upar_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_upar_n_4.check_status(agent_4_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_upar_pos))
				print("agent_4_pos: " + str(agent_4_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_upar_n_4.step_count)
				else:
					return (game_over_2,agents_comm_upar_n_4.step_count)

			print("Pred position before update: "+str(pred_pos))
			pred_pos = pred_upar.move_pred(graph_dict, agent_upar_pos, pred_pos)
			print("Pred position after update: "+str(pred_pos))
			#Check if any game-ending state has occured
			game_over_1 = agents_comm_upar_n_4.check_status(agent_upar_pos, pred_pos, prey_pos)
			game_over_2 = agents_comm_upar_n_4.check_status(agent_4_pos, pred_pos, prey_pos)
			#If game-ending condition reached, return the result and current count of steps
			if (game_over_1 != 0 or game_over_2 != 0):
				print("agent_ustr_pos: " + str(agent_upar_pos))
				print("agent_4_pos: " + str(agent_4_pos))
				print("prey_pos: " + str(prey_pos))
				print("pred_pos: " + str(pred_pos))
				if (game_over_1 != 0):
					return (game_over_1,agents_comm_upar_n_4.step_count)
				else:
					return (game_over_2,agents_comm_upar_n_4.step_count)
			agents_comm_upar_n_4.step_count += 1
		return (game_over,agents_comm_upar_n_4.step_count)


agnt_1_record = []
agnt_1_step_record = []
agnt_2_record = []
agnt_2_step_record = []
agnt_ustr_record = []
ustr_step_record = []
agnt_ustr_n_1_record = []
ustr_step_n_1_record = []
agnt_ustr_n_2_record = []
ustr_step_n_2_record = []
agnt_upar_n_3_record = []
upar_n_3_step_record = []
agnt_upar_n_4_record = []
upar_n_4_step_record = []
agnt_v_record = []
v_step_record = []
agnt_upar_record = []
upar_step_record = []
agnt_3_record = []
agnt_3_step_record = []
agnt_4_record = []
agnt_4_step_record = []

#Boolean falgs to run the agents
run_1 = False
run_2 = False
run_ustr = False
run_ustr_n_1 = False
run_ustr_n_2 = False
run_v = False
run_upar = False
run_3 = False
run_4 = False
run_upar_n_3 = False
run_upar_n_4 = False


##########AGENT 1 START##########
if run_1 == True:
	agnt_1_survivability = []
	agnt_1_final_fail = 0
	agnt_1_final_suspend = 0
	for i in range(100):
		graph1 = Graph()
		graph1.create_graph()
		for j in range(30):
			graph1.initialize_positions()
			agent_1 = Agent1()
			result = agent_1.proceed(graph1.graph_dict, graph1.connectivity_matrix, graph1.prey_pos, graph1.pred_pos, graph1.agent_pos)
			if result[0] == 0:
				print("Game suspended!")
				print("0")
				agnt_1_record.append(0)
				agnt_1_step_record.append(result[1])
			elif result[0] == 1:
				print("Agent 1 caught the Prey! Agent 1 won!")
				print("1")
				agnt_1_record.append(1)
				agnt_1_step_record.append(result[1])
			else:
				print("Predator caught the agent! Agent 1 lost!")
				print("2")
				agnt_1_record.append(2)
				agnt_1_step_record.append(result[1])
		agnt_1_sum_0 = 0
		agnt_1_sum_1 = 0
		agnt_1_sum_2 = 0

		for a in range(30):
			if agnt_1_record[(i*30)+a] == 0:
				agnt_1_sum_0 += 1
			elif agnt_1_record[(i*30)+a] == 1:
				agnt_1_sum_1 += 1
			elif agnt_1_record[(i*30)+a] == 2:
				agnt_1_sum_2 += 1
		agnt_1_final_suspend += agnt_1_sum_0
		agnt_1_final_fail += agnt_1_sum_2
		agnt_1_survivability.append((agnt_1_sum_1/30)*100)
	
	step_cnt_sum = 0
	for i in range(len(agnt_1_record)):
		print("Game result: " + str(agnt_1_record[i]) + ", step_count: " + str(agnt_1_step_record[i]))
		step_cnt_sum += agnt_1_step_record[i]
	print("Average Step Count for all " + str(x) + " games: " + str(step_cnt_sum/x))

	agnt_1_final_survivability = sum(agnt_1_survivability)/100
	print("agnt_1_survivability: " + str(agnt_1_survivability))
	
	print("Agent 1 Total iterations: " + str(len(agnt_1_record)))
	print("Agent 1 Count when Game Suspended: " + str(agnt_1_final_suspend))
	print("Agent 1 Count when Agent Lost: " + str(agnt_1_final_fail))
	print("Agent 1 Count when Agent Won: " + str(len(agnt_1_record) - (agnt_1_final_fail + agnt_1_final_suspend)))
	print("Agent 1 Survivability: " + str(agnt_1_final_survivability))
##########AGENT 1 END##########

##########AGENT 2 START##########
if run_2 == True:
	agnt_2_survivability = []
	agnt_2_final_fail = 0
	agnt_2_final_suspend = 0
	for i in range(100):
		graph_2 = Graph()
		graph_2.create_graph()
		for j in range(30):
			graph_2.initialize_positions()
			agent_2 = Agent2()
			result = agent_2.proceed(graph_2.graph_dict, graph_2.connectivity_matrix, graph_2.prey_pos, graph_2.pred_pos, graph_2.agent_pos)
			if result[0] == 0:
				print("Game suspended!")
				print("0")
				agnt_2_record.append(0)
				agnt_2_step_record.append(result[1])
			elif result[0] == 1:
				print("Agent 2 caught the Prey! Agent 2 won!")
				print("1")
				agnt_2_record.append(1)
				agnt_2_step_record.append(result[1])
			else:
				print("Predator caught the agent! Agent 2 lost!")
				print("2")
				agnt_2_record.append(2)
				agnt_2_step_record.append(result[1])
		agnt_2_sum_0 = 0
		agnt_2_sum_1 = 0
		agnt_2_sum_2 = 0

		for a in range(30):
			if agnt_2_record[(i*30)+a] == 0:
				agnt_2_sum_0 += 1
			elif agnt_2_record[(i*30)+a] == 1:
				agnt_2_sum_1 += 1
			elif agnt_2_record[(i*30)+a] == 2:
				agnt_2_sum_2 += 1
		agnt_2_final_suspend += agnt_2_sum_0
		agnt_2_final_fail += agnt_2_sum_2
		agnt_2_survivability.append((agnt_2_sum_1/30)*100)

	step_cnt_sum = 0
	for i in range(len(agnt_2_record)):
		print("Game result: " + str(agnt_2_record[i]) + ", step_count: " + str(agnt_2_step_record[i]))
		step_cnt_sum += agnt_2_step_record[i]
	print("Average Step Count for all " + str(x) + " games: " + str(step_cnt_sum/x))

	agnt_2_final_survivability = sum(agnt_2_survivability)/100
	print("agnt_2_survivability: " + str(agnt_2_survivability))
	
	print("Agent 2 Total iterations: " + str(len(agnt_2_record)))
	print("Agent 2 Count when Game Suspended: " + str(agnt_2_final_suspend))
	print("Agent 2 Count when Agent Lost: " + str(agnt_2_final_fail))
	print("Agent 2 Count when Agent Won: " + str(len(agnt_2_record) - (agnt_2_final_fail + agnt_2_final_suspend)))
	print("Agent 2 Survivability: " + str(agnt_2_final_survivability))
##########AGENT 2 END##########

##########AGENT 3 START##########
if run_3 == True:
	agnt_3_survivability = []
	agnt_3_final_fail = 0
	agnt_3_final_suspend = 0
	for i in range(100):
		graph3 = Graph()
		graph3.create_graph()
		for j in range(30):
			graph3.initialize_positions()
			agent_3 = Agent3()
			result = agent_3.proceed(graph3)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_3_record.append(0)
				agnt_3_step_record.append(result[1])
			elif result == 1:
				print("Agent 3 caught the Prey! Agent 3 won!")
				print("1")
				agnt_3_record.append(1)
				agnt_3_step_record.append(result[1])
			else:
				print("Predator caught the agent! Agent 3 lost!")
				print("2")
				agnt_3_record.append(2)
				agnt_3_step_record.append(result[1])
		agnt_3_sum_0 = 0
		agnt_3_sum_1 = 0
		agnt_3_sum_2 = 0

		for a in range(30):
			if agnt_3_record[(i*30)+a] == 0:
				agnt_3_sum_0 += 1
			elif agnt_3_record[(i*30)+a] == 1:
				agnt_3_sum_1 += 1
			elif agnt_3_record[(i*30)+a] == 2:
				agnt_3_sum_2 += 1
		agnt_3_final_suspend += agnt_3_sum_0
		agnt_3_final_fail += agnt_3_sum_2
		agnt_3_survivability.append((agnt_3_sum_1/30)*100)

	step_cnt_sum = 0
	for i in range(len(agnt_3_record)):
		print("Game result: " + str(agnt_3_record[i]) + ", step_count: " + str(agnt_3_step_record[i]))
		step_cnt_sum += agnt_3_step_record[i]
	print("Average Step Count for all " + str(x) + " games: " + str(step_cnt_sum/x))
	
	agnt_3_final_survivability = sum(agnt_3_survivability)/100
	print("agnt_3_survivability: " + str(agnt_3_survivability))
	
	print("Agent 3 Total iterations: " + str(len(agnt_3_record)))
	print("Agent 3 Count when Game Suspended: " + str(agnt_3_final_suspend))
	print("Agent 3 Count when Agent Lost: " + str(agnt_3_final_fail))
	print("Agent 3 Count when Agent Won: " + str(len(agnt_3_record) - (agnt_3_final_fail + agnt_3_final_suspend)))
	print("Agent 3 Survivability: " + str(agnt_3_final_survivability))
##########AGENT 3 END##########

##########AGENT 4 START##########
if run_4 == True:
	agnt_4_survivability = []
	agnt_4_final_fail = 0
	agnt_4_final_suspend = 0
	for i in range(100):
		graph_4 = Graph()
		graph_4.create_graph()
		for j in range(30):
			graph_4.initialize_positions()
			agent_4 = Agent4()
			result = agent_4.proceed(graph_4)
			if result == 0:
				print("Game suspended!")
				print("0")
				agnt_4_record.append(0)
				agnt_3_step_record.append(result[1])
			elif result == 1:
				print("Agent 4 caught the Prey! Agent 4 won!")
				print("1")
				agnt_4_record.append(1)
				agnt_4_step_record.append(result[1])
			else:
				print("Predator caught the agent! Agent 4 lost!")
				print("2")
				agnt_4_record.append(2)
				agnt_4_step_record.append(result[1])
		agnt_4_sum_0 = 0
		agnt_4_sum_1 = 0
		agnt_4_sum_2 = 0

		for a in range(30):
			if agnt_4_record[(i*30)+a] == 0:
				agnt_4_sum_0 += 1
			elif agnt_4_record[(i*30)+a] == 1:
				agnt_4_sum_1 += 1
			elif agnt_4_record[(i*30)+a] == 2:
				agnt_4_sum_2 += 1
		agnt_4_final_suspend += agnt_4_sum_0
		agnt_4_final_fail += agnt_4_sum_2
		agnt_4_survivability.append((agnt_4_sum_1/30)*100)

	step_cnt_sum = 0
	for i in range(len(agnt_4_record)):
		print("Game result: " + str(agnt_4_record[i]) + ", step_count: " + str(agnt_4_step_record[i]))
		step_cnt_sum += agnt_4_step_record[i]
	print("Average Step Count for all " + str(x) + " games: " + str(step_cnt_sum/x))

	agnt_4_final_survivability = sum(agnt_4_survivability)/100
	print("agnt_4_survivability: " + str(agnt_4_survivability))
	
	print("Agent 4 Total iterations: " + str(len(agnt_4_record)))
	print("Agent 4 Count when Game Suspended: " + str(agnt_4_final_suspend))
	print("Agent 4 Count when Agent Lost: " + str(agnt_4_final_fail))
	print("Agent 4 Count when Agent Won: " + str(len(agnt_4_record) - (agnt_4_final_fail + agnt_4_final_suspend)))
	print("Agent 4 Survivability: " + str(agnt_4_final_survivability))
##########AGENT 4 END##########

##########AGENT U Star START##########
if run_ustr == True:
	agnt_ustr_survivability = []
	agnt_ustr_final_fail = 0
	agnt_ustr_final_suspend = 0
	x = 30
	for i in range(1):#00):
		agents_comm_ustr = agents_common()
		graph_ustr = Graph()
		graph_ustr.create_graph()
		agents_comm_ustr.initialize_vals(graph_ustr.graph_dict)
		print("utility_vals: " + str(agents_comm_ustr.utility_vals))
		max_val = -1
		min_val = 10000
		max_val_key = (0,0,0)
		min_val_key = (0,0,0)
		for key in agents_comm_ustr.utility_vals.keys():
			if agents_comm_ustr.utility_vals[key] > max_val:
				max_val = agents_comm_ustr.utility_vals[key]
				max_val_key = key
			if agents_comm_ustr.utility_vals[key] < min_val:
				min_val = agents_comm_ustr.utility_vals[key]
				min_val_key = key

		print("max Ustr val: " + str(max_val))
		print("max Ustr val key: " + str(max_val_key))
		print("min Ustr val: " + str(min_val))
		print("min Ustr val key: " + str(min_val_key))
		print("Completed")
		for j in range(x):
			graph_ustr.initialize_positions()
			agent_ustr = Agent_u_star()
			result = agent_ustr.proceed(graph_ustr.graph_dict, graph_ustr.prey_pos, graph_ustr.pred_pos, graph_ustr.agent_pos, agents_comm_ustr)
			if result[0] == 0:
				print("Game suspended!")
				print("0")
				agnt_ustr_record.append(0)
				ustr_step_record.append(result[1])
			elif result[0] == 1:
				print("Agent U star caught the Prey! Agent U star won!")
				print("1")
				agnt_ustr_record.append(1)
				ustr_step_record.append(result[1])
			else:
				print("Predator caught the agent! Agent U star lost!")
				print("2")
				agnt_ustr_record.append(2)
				ustr_step_record.append(result[1])

		agnt_ustr_sum_0 = 0
		agnt_ustr_sum_1 = 0
		agnt_ustr_sum_2 = 0
	
		for a in range(30):
			if agnt_ustr_record[(i*30)+a] == 0:
				agnt_ustr_sum_0 += 1
			elif agnt_ustr_record[(i*30)+a] == 1:
				agnt_ustr_sum_1 += 1
			elif agnt_ustr_record[(i*30)+a] == 2:
				agnt_ustr_sum_2 += 1
		agnt_ustr_final_suspend += agnt_ustr_sum_0
		agnt_ustr_final_fail += agnt_ustr_sum_2
		agnt_ustr_survivability.append((agnt_ustr_sum_1/30)*100)
	
	step_cnt_sum = 0
	for i in range(len(agnt_ustr_record)):
		print("Game result: " + str(agnt_ustr_record[i]) + ", step_count: " + str(ustr_step_record[i]))
		step_cnt_sum += ustr_step_record[i]
	print("Average Step Count for all " + str(x) + " games: " + str(step_cnt_sum/x))


	agnt_ustr_final_survivability = sum(agnt_ustr_survivability)#/100
	print("agnt_ustr_survivability: " + str(agnt_ustr_survivability))
	
	print("Agent ustr Total iterations: " + str(len(agnt_ustr_record)))
	print("Agent ustr Count when Game Suspended: " + str(agnt_ustr_final_suspend))
	print("Agent ustr Count when Agent Lost: " + str(agnt_ustr_final_fail))
	print("Agent ustr Count when Agent Won: " + str(len(agnt_ustr_record) - (agnt_ustr_final_fail + agnt_ustr_final_suspend)))
	print("Agent ustr Survivability: " + str(agnt_ustr_final_survivability))
##########AGENT U Star END##########

##########AGENT U Partial START##########
if run_upar == True:
	agnt_upar_survivability = []
	agnt_upar_final_fail = 0
	agnt_upar_final_suspend = 0
	x = 100
	for i in range(30):#00):
		agents_comm_upar = agents_common()
		graph_upar = Graph()
		graph_upar.create_graph()
		agents_comm_upar.initialize_vals(graph_upar.graph_dict)
		print("graph: " + str(graph_upar.graph_dict))
		print("utility_vals: " + str(agents_comm_upar.utility_vals))
		for j in range(x):
			graph_upar.initialize_positions()
			agent_upar = Agent_u_partial()
			result = agent_upar.proceed(graph_upar.graph_dict, graph_upar.prey_pos, graph_upar.pred_pos, graph_upar.agent_pos, agents_comm_upar)
			if result[0] == 0:
				print("Game suspended!")
				print("0")
				agnt_upar_record.append(0)
				upar_step_record.append(result[1])
			elif result[0] == 1:
				print("Agent U partial caught the Prey! Agent U partial won!")
				print("1")
				agnt_upar_record.append(1)
				upar_step_record.append(result[1])
			else:
				print("Predator caught the agent! Agent U partial lost!")
				print("2")
				agnt_upar_record.append(2)
				upar_step_record.append(result[1])

		agnt_upar_sum_0 = 0
		agnt_upar_sum_1 = 0
		agnt_upar_sum_2 = 0
	
		for a in range(x):
			if agnt_upar_record[(i*x)+a] == 0:
				agnt_upar_sum_0 += 1
			elif agnt_upar_record[(i*x)+a] == 1:
				agnt_upar_sum_1 += 1
			elif agnt_upar_record[(i*x)+a] == 2:
				agnt_upar_sum_2 += 1
		agnt_upar_final_suspend += agnt_upar_sum_0
		agnt_upar_final_fail += agnt_upar_sum_2
		agnt_upar_survivability.append((agnt_upar_sum_1/x)*100)
	
	step_cnt_sum = 0
	for i in range(len(agnt_upar_record)):
		print("Game result: " + str(agnt_upar_record[i]) + ", step_count: " + str(upar_step_record[i]))
		step_cnt_sum += upar_step_record[i]
	print("Average Step Count for all " + str(x) + " games: " + str(step_cnt_sum/x))

	agnt_upar_final_survivability = sum(agnt_upar_survivability)#/100
	print("agnt_upar_survivability: " + str(agnt_upar_survivability))
	
	print("Agent upar Total iterations: " + str(len(agnt_upar_record)))
	print("Agent upar Count when Game Suspended: " + str(agnt_upar_final_suspend))
	print("Agent upar Count when Agent Lost: " + str(agnt_upar_final_fail))
	print("Agent upar Count when Agent Won: " + str(len(agnt_upar_record) - (agnt_upar_final_fail + agnt_upar_final_suspend)))
	print("Agent upar Survivability: " + str(agnt_upar_final_survivability))
##########AGENT U Partial END##########

##########AGENT V Partial START##########
if run_v == True:
	agnt_v_survivability = []
	agnt_v_final_fail = 0
	agnt_v_final_suspend = 0
	x = 100
	for i in range(1):#
		agents_comm_v = agents_common()
		graph_v = Graph()
		graph_v.create_graph()
		agents_comm_v.initialize_vals(graph_v.graph_dict)
		nn = neuralnet()
		nn.calculate_utils(graph_v, agents_comm_v)
		for j in range(x):
			graph_v.initialize_positions()
			agent_v = Agent_v()
			result = agent_v.proceed(graph_v.graph_dict, graph_v.prey_pos, graph_v.pred_pos, graph_v.agent_pos, agents_comm_v, nn)
			if result[0] == 0:
				print("Game suspended!")
				print("0")
				agnt_v_record.append(0)
				v_step_record.append(result[1])
			elif result[0] == 1:
				print("Agent V partial caught the Prey! Agent V won!")
				print("1")
				agnt_v_record.append(1)
				v_step_record.append(result[1])
			else:
				print("Predator caught the agent! Agent V lost!")
				print("2")
				agnt_v_record.append(2)
				v_step_record.append(result[1])

		agnt_v_sum_0 = 0
		agnt_v_sum_1 = 0
		agnt_v_sum_2 = 0
	
		for a in range(x):
			if agnt_v_record[(i*x)+a] == 0:
				agnt_v_sum_0 += 1
			elif agnt_v_record[(i*x)+a] == 1:
				agnt_v_sum_1 += 1
			elif agnt_v_record[(i*x)+a] == 2:
				agnt_v_sum_2 += 1
		agnt_v_final_suspend += agnt_v_sum_0
		agnt_v_final_fail += agnt_v_sum_2
		agnt_v_survivability.append((agnt_v_sum_1/x)*100)

	step_cnt_sum = 0
	for i in range(len(agnt_v_record)):
		print("Game result: " + str(agnt_v_record[i]) + ", step_count: " + str(v_step_record[i]))
		step_cnt_sum += v_step_record[i]
	print("Average Step Count for all " + str(x) + " games: " + str(step_cnt_sum/x))
		
	agnt_v_final_survivability = sum(agnt_v_survivability)#/100
	print("agnt_v_survivability: " + str(agnt_v_survivability))
	
	print("Agent v Total iterations: " + str(len(agnt_v_record)))
	print("Agent v Count when Game Suspended: " + str(agnt_v_final_suspend))
	print("Agent v Count when Agent Lost: " + str(agnt_v_final_fail))
	print("Agent v Count when Agent Won: " + str(len(agnt_v_record) - (agnt_v_final_fail + agnt_v_final_suspend)))
	print("Agent v Survivability: " + str(agnt_v_final_survivability))
##########AGENT V END##########

##########AGENT U Star and 1 START##########
if run_ustr_n_1 == True:
	agnt_ustr_n_1_survivability = []
	agnt_ustr_n_1_final_fail = 0
	agnt_ustr_n_1_final_suspend = 0
	x = 100
	for i in range(1):
		agents_comm_ustr_n_1 = agents_common()
		graph_ustr_n_1 = Graph()
		graph_ustr_n_1.create_graph()
		agents_comm_ustr_n_1.initialize_vals(graph_ustr_n_1.graph_dict)
		print("graph_ustr_n_1.graph_dict: " + str(graph_ustr_n_1.graph_dict))
		for j in range(x):
			agent_1 = Agent1()
			graph_ustr_n_1.initialize_positions()
			agent_ustr_n_1 = Agent_u_star_n_1()
			result = agent_ustr_n_1.proceed(graph_ustr_n_1.graph_dict, graph_ustr_n_1.prey_pos, graph_ustr_n_1.pred_pos, graph_ustr_n_1.agent_pos, agents_comm_ustr_n_1, agent_1)
			if result[0] == 0:
				print("Game suspended!")
				print("0")
				agnt_ustr_n_1_record.append(0)
				ustr_step_n_1_record.append(result[1])
			elif (result[0] == 1):
				print("Some caught the Prey! Some agent won!")
				agnt_ustr_n_1_record.append(1)
				ustr_step_n_1_record.append(result[1])
			else:
				print("Predator caught some agent! Some agent lost!")
				agnt_ustr_n_1_record.append(2)
				ustr_step_n_1_record.append(result[1])
##########AGENT U Star and 1 END##########

##########AGENT U Star and 2 START##########
if run_ustr_n_2 == True:
	agnt_ustr_n_2_survivability = []
	agnt_ustr_n_2_final_fail = 0
	agnt_ustr_n_2_final_suspend = 0
	x = 100
	for i in range(1):
		agents_comm_ustr_n_2 = agents_common()
		graph_ustr_n_2 = Graph()
		graph_ustr_n_2.create_graph()
		agents_comm_ustr_n_2.initialize_vals(graph_ustr_n_2.graph_dict)
		print("graph_ustr_n_2.graph_dict: " + str(graph_ustr_n_2.graph_dict))
		for j in range(x):
			agent_2 = Agent2()
			graph_ustr_n_2.initialize_positions()
			agent_ustr_n_2 = Agent_u_star_n_2()
			result = agent_ustr_n_2.proceed(graph_ustr_n_2.graph_dict, graph_ustr_n_2.prey_pos, graph_ustr_n_2.pred_pos, graph_ustr_n_2.agent_pos, agents_comm_ustr_n_2, agent_2)
			if result[0] == 0:
				print("Game suspended!")
				print("0")
				agnt_ustr_n_2_record.append(0)
				ustr_step_n_1_record.append(result[1])
			elif (result[0] == 1):
				print("Some caught the Prey! Some agent won!")
				agnt_ustr_n_2_record.append(1)
				ustr_step_n_1_record.append(result[1])
			else:
				print("Predator caught some agent! Some agent lost!")
				agnt_ustr_n_2_record.append(2)
				ustr_step_n_1_record.append(result[1])
##########AGENT U Star and 2 END##########


##########AGENT U Partial and 3 START##########
if run_upar_n_3 == True:
	agnt_upar_n_3_survivability = []
	agnt_upar_n_3_final_fail = 0
	agnt_upar_n_3_final_suspend = 0
	x = 30
	for i in range(1):#00):
		agents_comm_upar_n_3 = agents_common()
		graph_upar_n_3 = Graph()
		graph_upar_n_3.create_graph()
		agents_comm_upar_n_3.initialize_vals(graph_upar_n_3.graph_dict)
		print("graph: " + str(graph_upar_n_3.graph_dict))
		print("utility_vals: " + str(agents_comm_upar_n_3.utility_vals))
		for j in range(x):
			graph_upar_n_3.initialize_positions()
			agent_upar_n_3 = Agent_u_partial_n_3()
			result = agent_upar_n_3.proceed(graph_upar_n_3.graph_dict, graph_upar_n_3.prey_pos, graph_upar_n_3.pred_pos, graph_upar_n_3.agent_pos, agents_comm_upar_n_3)
			if result[0] == 0:
				print("Game suspended!")
				print("0")
				agnt_upar_n_3_record.append(0)
				upar_n_3_step_record.append(result[1])
			elif result[0] == 1:
				print("Agent U partial caught the Prey! Agent U partial won!")
				print("1")
				agnt_upar_n_3_record.append(1)
				upar_n_3_step_record.append(result[1])
			else:
				print("Predator caught the agent! Agent U partial lost!")
				print("2")
				agnt_upar_n_3_record.append(2)
				upar_n_3_step_record.append(result[1])
##########AGENT U Partial and 3 END##########

##########AGENT U Partial and 4 START##########
if run_upar_n_4 == True:
	agnt_upar_n_4_survivability = []
	agnt_upar_n_4_final_fail = 0
	agnt_upar_n_4_final_suspend = 0
	x = 1
	for i in range(1):
		agents_comm_upar_n_4 = agents_common()
		graph_upar_n_4 = Graph()
		graph_upar_n_4.create_graph()
		agents_comm_upar_n_4.initialize_vals(graph_upar_n_4.graph_dict)
		print("graph: " + str(graph_upar_n_4.graph_dict))
		print("utility_vals: " + str(agents_comm_upar_n_4.utility_vals))
		for j in range(x):
			graph_upar_n_4.initialize_positions()
			agent_upar_n_4 = Agent_u_partial_n_4()
			result = agent_upar_n_4.proceed(graph_upar_n_4.graph_dict, graph_upar_n_4.prey_pos, graph_upar_n_4.pred_pos, graph_upar_n_4.agent_pos, agents_comm_upar_n_4)
			if result[0] == 0:
				print("Game suspended!")
				print("0")
				agnt_upar_n_4_record.append(0)
				upar_n_4_step_record.append(result[1])
			elif result[0] == 1:
				print("Agent U partial caught the Prey! Agent U partial won!")
				print("1")
				agnt_upar_n_4_record.append(1)
				upar_n_4_step_record.append(result[1])
			else:
				print("Predator caught the agent! Agent U partial lost!")
				print("2")
				agnt_upar_n_4_record.append(2)
				upar_n_4_step_record.append(result[1])
##########AGENT U Partial and 4 END##########