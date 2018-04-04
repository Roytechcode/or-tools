# This Python file uses the following encoding: utf-8
# Copyright 2015 Tin Arm Engineering AB
# Copyright 2017 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vehicle Routing Problem with Time Windows.

   This is a sample using the routing library python wrapper to solve a
   VRP problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.
   Distances are computed using the Manhattan distances. Distances are in km.

   The optimization engine uses local search to improve solutions, first
   solutions being generated using a cheapest addition heuristic.
"""

from __future__ import print_function
import sys
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# Problem Data Definition
class DataProblem():
    """Stores the data for the problem"""
    def __init__(self):
        """Initializes the data for the problem"""
        self._num_vehicles = 5

        self._locations = \
            [[82, 76], [96, 44], [50, 5], [49, 8], [13, 7], [29, 89], [58, 30], [84, 39],
             [14, 24], [12, 39], [3, 82], [5, 10], [98, 52], [84, 25], [61, 59], [1, 65],
             [88, 51], [91, 2], [19, 32], [93, 3], [50, 93], [98, 14], [5, 42], [42, 9],
             [61, 62], [9, 97], [80, 55], [57, 69], [23, 15], [20, 70], [85, 60], [98, 5]]
        self._depot = 0

        # Check data coherency
        if self.num_locations == 0:
            raise ValueError('Locations must be greater than 0.')

    @property
    def num_vehicles(self):
        """Gets number of vehicles"""
        return self._num_vehicles

    @property
    def locations(self):
        """Gets locations"""
        return self._locations

    @property
    def num_locations(self):
        """Gets number of locations"""
        return len(self.locations)

    def manhattan_distance(self, from_node, to_node):
        """Computes the Manhattan distance between two nodes"""
        return (abs(self.locations[from_node][0] - self.locations[to_node][0]) +
                abs(self.locations[from_node][1] - self.locations[to_node][1]))

    @property
    def depot(self):
        """Gets depot location index"""
        return self._depot

# Distance callback
class CreateDistanceCallback(object): # pylint: disable=too-few-public-methods
    """Creates callback to return distance between points."""
    def __init__(self, data):
        """Initializes the distance matrix."""
        self._distance = {}

        # precompute distance between location to have distance callback in O(1)
        for from_node in xrange(data.num_locations):
            self._distance[from_node] = {}
            for to_node in xrange(data.num_locations):
                if from_node == to_node:
                    self._distance[from_node][to_node] = 0
                else:
                    self._distance[from_node][to_node] = (
                        data.manhattan_distance(from_node, to_node))

    def distance(self, from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return self._distance[from_node][to_node]


def print_assignment(data, routing, assignment):
    """Prints solution"""
    # Solution cost.
    print("Total distance of all routes: {0}\n".format(assignment.ObjectiveValue()))
    # Inspect solution.
    for vehicle_id in xrange(data.num_vehicles):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
        route_dist = 0

        while not routing.IsEnd(index):
            node_index = routing.IndexToNode(index)
            next_node_index = routing.IndexToNode(assignment.Value(routing.NextVar(index)))
            route_dist += data.manhattan_distance(node_index, next_node_index)
            plan_output += ' {node_index} -> '.format(node_index=node_index)
            index = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)
        plan_output += ' {node_index}\n'.format(node_index=node_index)
        plan_output += 'Distance of the route {0}: {dist}\n'.format(
            vehicle_id,
            dist=route_dist)
        print(plan_output, '\n')

def main():
    """Entry point of the program"""
    # Instanciate the data problem.
    data = DataProblem()

    # Create routing model.
    # The number of nodes of the VRP is num_locations.
    # Nodes are indexed from 0 to num_locations - 1.
    # By default the start of a route is node 0.
    routing = pywrapcp.RoutingModel(
        data.num_locations,
        data.num_vehicles,
        data.depot)

    # Adding the custom distance function.
    dist_callback = CreateDistanceCallback(data).distance
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

    # Setting first solution heuristic (cheapest addition).
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Display a solution if any.
    if assignment:
        print_assignment(data, routing, assignment)
    else:
        print('No solution found.')
        sys.exit(2)

if __name__ == '__main__':
    main()
