# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:46:37 2021

@author: trymlind
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

class Paper(object):
    def __init__(self):
        self.publication_year = int(2020 - np.random.exponential(20))
        
        # center of distribution taken from:
        # https://www.researchtrends.com/issue-32-march-2013/citation-characteristics-in-the-arts-humanities-2/
        # SD is arbitrarily chosen
        self.number_of_out_citations = np.random.normal(54, 20)
        self.out_citations = []
        self.in_citations = []
        
    def set_out_citation(self, the_scientific_literature):
        permuted_litterature_indexes = np.random.permutation(len(the_scientific_litterature))
        
        current_index = 0
        attempts = 0
        while len(self.out_citations) < self.number_of_out_citations and attempts <= 5:
            #print("finding out citations")
            out_citation_index = permuted_litterature_indexes[current_index]
            potential_out_citation = the_scientific_litterature[out_citation_index]
            
            if potential_out_citation not in self.out_citations:
                if self.out_citation_probability(potential_out_citation):
                    self.out_citations.append(potential_out_citation)
                    potential_out_citation.in_citations.append(self)
                    print("connected! ", len(self.out_citations))
            current_index += 1
            
            if current_index >= len(the_scientific_litterature):
                current_index = 0
                attempts += 1
        
    def out_citation_probability(self, potential_out_citation):
        potential_out_citation.publication_year
        # most papers cite other papers that are close to themselves (but obviously published earlier)
        if potential_out_citation.publication_year <= self.publication_year:
            publication_year_distance = self.publication_year - potential_out_citation.publication_year
            
            # calculate the probability of of the potential out-citation being cited based on the gamma distribution
            probability_threshold = stats.gamma.pdf(publication_year_distance, a= 2, loc = 0.1, scale = 2)
            
            # if the publication has similar out-citations to the current publication it is more likely to be in the same field
            print()
            print(self.publication_year, potential_out_citation.publication_year)
            print(probability_threshold)
            for paper in self.out_citations:
                if paper in potential_out_citation.out_citations:
                    probability_threshold += (1 - probability_threshold) * (1/54)
            print(probability_threshold)
            random_float = np.random.rand(1)
            print(random_float)
            if random_float < probability_threshold:
                return True
            else:
                return False
            
        else:
            return False
            
# Generate the scientific litterature
nr_of_publications = 100

the_scientific_litterature = []
for _ in range(nr_of_publications):
    the_scientific_litterature.append(Paper())

# Connect papers
nr = 1
for paper in the_scientific_litterature:
    paper.set_out_citation(the_scientific_litterature)
    print(nr)
    nr += 1

def repulsive_force(distance, force_max):
    return -force_max/(distance + 0.0000001) #np.exp(-distance) * force_max

def attractive_force(distance, spring_stiffness):
    # hookes law
    return -spring_stiffness * distance


paper = the_scientific_litterature[np.random.randint(len(the_scientific_litterature))]
image = np.zeros((1000,1000))
image_x = image.shape[0]
image_y = image.shape[1]

margins = 20
year_scale = (image_y - margins) /(2020 - paper.publication_year)
root_node_year = paper.publication_year

root_node_position = np.array([int(image_y-margins), int(image_x/2)])
positions_array = np.array([[image_y-margins], [image_x/2]])

node_velocities = np.array([[0.], [0.]])

repulsive_force_max = 5
spring_tightness = 0.00001


current_node = 0
graph_nodes = [paper]
total_nodes = len(graph_nodes)
connections = np.zeros((total_nodes, total_nodes))

breaker = False

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (image_y ,image_x))


while len(graph_nodes[current_node].in_citations) > 0 and current_node <= len(graph_nodes):
    
    for paper in graph_nodes[current_node].in_citations:
        print(current_node, graph_nodes[current_node].publication_year, paper.publication_year )
        #print('\n start \n')
        #print(paper.publication_year)
        #print(total_nodes)
        #print(paper.publication_year)
        graph_nodes.append(paper)
        
        total_nodes = len(graph_nodes)
        #print(total_nodes)
        #print(positions_array)
        
        node_x = positions_array[1,current_node]
        node_y = image_y-1 - (paper.publication_year - root_node_year)*year_scale
        node_position = np.array([[node_y], [node_x]])
        #print(node_position)
        positions_array = np.concatenate((positions_array, node_position), axis = 1)
        
        
        new_node_velocity = np.array([[0],[0]])
        node_velocities = np.concatenate((node_velocities, new_node_velocity), axis = 1)
        #print("np ", node_positions)
        
   
        
        #positions_array = np.rot90(positions_array, axes = (0,1))
        #print("\n pa \n", positions_array, '\n')
        positions_square = np.repeat(positions_array[:,:,np.newaxis], positions_array.shape[1], axis = 2)
        positions_rotated = np.rot90(positions_square, axes = (1,2)) 
        
        #print(positions_square)
        distance_vectors = positions_rotated - positions_square
        distance_magnitudes = np.linalg.norm(distance_vectors, axis = 0, ord = 2)
    
        new_connections = np.zeros((total_nodes, total_nodes))
        new_connections[:total_nodes-1, :total_nodes-1] = connections
        new_connections[current_node, total_nodes-1] = 1
        new_connections[total_nodes-1, current_node] = 1
        connections = new_connections
        
        repulsion_distance_magnitude = np.zeros(distance_magnitudes.shape)
        repulsion_distance_magnitude[:,:] = distance_magnitudes[:,:] * np.abs(distance_vectors[0,:,:])
        mid = np.arange(repulsion_distance_magnitude.shape[0])
        repulsion_distance_magnitude[mid,mid] = 200
        repulsive_force_strength = repulsive_force(repulsion_distance_magnitude, repulsive_force_max)
        #print('\n \n \n', repulsion_distance_magnitude, '\n \n', repulsive_force_strength)
        min_pos = ind = np.unravel_index(np.argmin(repulsive_force_strength, axis=None), repulsive_force_strength.shape)
        #print(np.amax(repulsive_force_strength), repulsion_distance_magnitude[min_pos[0], min_pos[1]])
        repulsive_force_strength =  np.repeat(repulsive_force_strength[np.newaxis,:,:], 2, axis = 0)
        
        attractive_force_strength = attractive_force(distance_magnitudes, spring_tightness)
        attractive_force_strength*= connections
        attractive_force_strength = np.repeat(attractive_force_strength[np.newaxis,:,:], 2, axis = 0)
        #print(positions_array.shape)
        #print("\n afs \n ",attractive_force_strength)
        #print("c \n", connections)
        #attractive_force_strength *= connections

        distance_magnitudes_2 = np.repeat(distance_magnitudes[np.newaxis,:,:], 2, axis = 0)
        unit_distance_vectors =  distance_vectors / (distance_magnitudes_2 + 0.00000001)


        
        repulsive_forces = unit_distance_vectors * repulsive_force_strength
        attractive_forces = unit_distance_vectors * attractive_force_strength

        #uncertain if axis is correct
        total_forces = np.sum(attractive_forces, axis = 1)
        
        if total_nodes > 1:
            positions_array[0,-1] += np.random.randint(-10,10)
            positions_array[1,-1] += np.random.randint(-200,200)
        
        #print("total forces ", total_forces.shape)
        #print("attractive forces ", attractive_forces.shape)
        node_velocities*= 0
        node_velocities[1,:] += total_forces[1,:]
        #positions_array[1,:] += node_velocities[1,:]
        
        #print(np.amax(total_forces[1,:]))
        #print('\n \n \n', distance_magnitudes[0,1],'\n\n', attractive_force_strength[0,1], '\n\n', unit_distance_vectors[:,0,1])
        #print("error?")
        #print(positions_array.shape)
        
        
        above_image_x = positions_array[1,:] > image_x
        below_image_x = positions_array[1,:] < 0
        
        positions_array[1,:] *= above_image_x == 0
        positions_array[1,:] += above_image_x*(image_x-1)
        
        positions_array[1,:] *= below_image_x == 0
        
        positions_array[:,0] = root_node_position
        image = np.zeros((image_x, image_y, 3))
        image[positions_array[0,:].astype(np.int32), positions_array[1,:].astype(np.int32), :] = 255
        
        connection_lines = np.where(connections == 1)
        connection_lines = np.array(connection_lines)
        
        for index in range(np.shape(connection_lines)[1]):
                start_point = connection_lines[0, index]
                end_point = connection_lines[1, index]
                
                start_point = positions_array[:,start_point].astype(np.int32)
                end_point = positions_array[:,end_point].astype(np.int32)
                
                start_point = np.flip(start_point)
                end_point = np.flip(end_point)
                
                start_point = tuple(start_point)
                end_point = tuple(end_point)
                
                print(start_point, end_point)
                cv2.line(image, start_point, end_point, color = (100,0,0), thickness = 1)
                cv2.circle(image, end_point, 3, [100,100,100], 1)

        cv2.imshow('frame', image)
        out.write(image)

        #time.sleep(0.1)
        if len(graph_nodes) > 4000:
            breaker = True
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            breaker = True
            break
    current_node += 1
    if breaker:
        break
'''

while True:   
    cv2.imshow('frame', image)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''        
cv2.destroyAllWindows()

        