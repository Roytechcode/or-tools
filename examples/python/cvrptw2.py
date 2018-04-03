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

"""Capacitated Vehicle Routing Problem with Time Windows.

   This is a sample using the routing library python wrapper to solve a
   CVRPTW problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.
   The variant which is tackled by this model includes a capacity dimension
   and time windows.
   Distances are computed using the Manhattan distances. Distances are in km
   and times in seconds.

   The optimization engine uses local search to improve solutions, first
   solutions being generated using a cheapest addition heuristic.
"""

from __future__ import print_function
import sys
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# Problem Data Definition
class CreateDataProblem(object):
    """Stores the data for the problem"""
    def __init__(self):
        """Initializes the data for the problem"""
        self._num_vehicles = 5

        self._depot = 0

        self._locations = \
            [[82, 76], [96, 44], [50, 5], [49, 8], [13, 7], [29, 89], [58, 30], [84, 39],
             [14, 24], [12, 39], [3, 82], [5, 10], [98, 52], [84, 25], [61, 59], [1, 65],
             [88, 51], [91, 2], [19, 32], [93, 3], [50, 93], [98, 14], [5, 42], [42, 9],
             [61, 62], [9, 97], [80, 55], [57, 69], [23, 15], [20, 70], [85, 60], [98, 5]]

        self._num_locations = len(self._locations)

        # Capacity related constraints
        self._vehicle_capacity = 100

        self._demands = \
            [0, 19, 21, 6, 19, 7, 12, 16,
             6, 16, 8, 14, 21, 16, 3, 22,
             18, 19, 1, 24, 8, 12, 4, 8,
             24, 24, 2, 20, 15, 2, 14, 9]

        # Time related constraints
        # Travel speed: 80km/h to convert in km/s
        self._vehicle_speed = 80 / 3600.

        # Time to deliver a package to a customer: 3min/unit
        self._time_per_demand_unit = 3 * 60

        self._start_times = \
            [0, 5080, 1030, 4930, 2250, 5310, 890, 5650,
             5400, 1080, 6020, 4660, 3560, 3030, 3990, 3820,
             3620, 5210, 230, 4890, 4450, 3180, 3800, 550,
             5740, 5150, 1100, 3100, 3870, 4910, 3280, 730]

        # The width of the time window: 5 hours.
        self._tw_duration = 5 * 60 * 60

        # In this example, the time window widths is the same at each location, so we define the end
        # times to be start times + tw_duration.
        # For problems in which the time window widths vary by location, you can explicitly define
        # the list of end_times, as we have done for start_times.
        self._end_times = [start + self._tw_duration for start in self._start_times]


        # Check data coherency
        if (len(self._locations) != len(self._demands) or
                len(self._locations) != len(self._start_times) or
                len(self._locations) != len(self._end_times)):
            raise RuntimeError("Inconsistent data problem!")

    @property
    def num_vehicles(self):
        """Gets number of vehicles"""
        return self._num_vehicles

    @property
    def depot(self):
        """Gets depot location index"""
        return self._depot

    @property
    def locations(self):
        """Gets locations"""
        return self._locations

    @property
    def num_locations(self):
        """Gets number of locations"""
        return self._num_locations

    def manhattan_distance(self, from_node, to_node):
        """Computes the Manhattan distance between from_node and to_node"""
        return abs(self.locations[from_node][0] - self.locations[to_node][0]) + \
            abs(self.locations[from_node][1] - self.locations[to_node][1])

    @property
    def vehicle_capacity(self):
        """Gets vehicle capacity"""
        return self._vehicle_capacity

    @property
    def demands(self):
        """Gets demands"""
        return self._demands

    @property
    def vehicle_speed(self):
        """Gets the average travel speed of a vehicle"""
        return self._vehicle_speed

    @property
    def time_per_demand_unit(self):
        """Gets the average time per demand unit"""
        return self._time_per_demand_unit

    @property
    def start_times(self):
        """Gets start times"""
        return self._start_times

    @property
    def end_times(self):
        """Gets end times"""
        return self._end_times

# Distance callback
class CreateDistanceCallback(object): # pylint: disable=too-few-public-methods
    """Creates callback to return distance between points."""

    def __init__(self, data):
        """Initializes the distance matrix."""
        self.matrix = {}

        for from_node in xrange(data.num_locations):
            self.matrix[from_node] = {}
            for to_node in xrange(data.num_locations):
                if from_node == to_node:
                    self.matrix[from_node][to_node] = 0
                else:
                    self.matrix[from_node][to_node] = data.manhattan_distance(from_node, to_node)

    def distance(self, from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return self.matrix[from_node][to_node]


# Demand callback
class CreateDemandCallback(object): # pylint: disable=too-few-public-methods
    """Creates callback to get demands at each location."""

    def __init__(self, data):
        """Initializes the demand array."""
        self._demands = data.demands

    def demand(self, from_node, to_node):
        """Returns the demand of the current node"""
        del to_node
        return self._demands[from_node]

# Time callback (equals service time plus travel time).
class CreateTimeCallback(object):
    """Creates callback to get total times between locations."""
    @staticmethod
    def service_time(data, node):
        """Gets the service time for the specified location."""
        return data.demands[node] * data.time_per_demand_unit

    @staticmethod
    def travel_time(data, from_node, to_node):
        """Gets the travel times between two locations."""
        if from_node == to_node:
            travel_time = 0
        else:
            travel_time = data.manhattan_distance(from_node, to_node) / data.vehicle_speed
        return travel_time

    def __init__(self, data):
        """Initializes the total time matrix."""
        self._total_time = {}

        for from_node in xrange(data.num_locations):
            self._total_time[from_node] = {}
            for to_node in xrange(data.num_locations):
                if from_node == to_node:
                    self._total_time[from_node][to_node] = 0
                else:
                    self._total_time[from_node][to_node] = \
                        self.service_time(data, from_node) + \
                        self.travel_time(data, from_node, to_node)

    def time(self, from_node, to_node):
        """Returns the total time between the two nodes"""
        return self._total_time[from_node][to_node]

def print_assignment(data, routing, assignment, capacity, time):
    """Prints solution"""
    # Solution cost.
    print("Total distance of all routes: {0}\n".format(assignment.ObjectiveValue()))
    # Inspect solution.
    capacity_dimension = routing.GetDimensionOrDie(capacity)
    time_dimension = routing.GetDimensionOrDie(time)

    for vehicle_id in xrange(data.num_vehicles):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
        route_dist = 0

        while not routing.IsEnd(index):
            node_index = routing.IndexToNode(index)
            next_node_index = routing.IndexToNode(assignment.Value(routing.NextVar(index)))
            route_dist += data.manhattan_distance(node_index, next_node_index)
            load_var = capacity_dimension.CumulVar(index)
            time_var = time_dimension.CumulVar(index)
            plan_output += ' {node_index} Load({load}) Time({tmin}, {tmax}) -> '.format(
                node_index=node_index,
                load=assignment.Value(load_var),
                tmin=str(assignment.Min(time_var)),
                tmax=str(assignment.Max(time_var)))
            index = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)
        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)
        plan_output += ' {node_index} Load({load}) Time({tmin}, {tmax})\n'.format(
            node_index=node_index,
            load=assignment.Value(load_var),
            tmin=str(assignment.Min(time_var)),
            tmax=str(assignment.Max(time_var)))
        plan_output += 'Distance of the route {0}: {dist}\n'.format(
            vehicle_id,
            dist=route_dist)
        plan_output += 'Demand met by vehicle {0}: {load}\n'.format(
            vehicle_id,
            load=assignment.Value(load_var))
        print(plan_output, '\n')


def main():
    """Entry point of the program"""
    # Create the data.
    data = CreateDataProblem()

    if data.num_locations == 0:
        raise ValueError('Specify an instance greater than 0.')

    # Create routing model.
    # The number of nodes of the VRP is num_locations.
    # Nodes are indexed from 0 to num_locations - 1.
    # By default the start of a route is node 0.
    routing = pywrapcp.RoutingModel(data.num_locations, data.num_vehicles, data.depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    # Setting first solution heuristic (cheapest addition).
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Callbacks to the distance function.
    dist_between_locations = CreateDistanceCallback(data)
    dist_callback = dist_between_locations.distance
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

    # Adding a Capacity dimension constraints.
    demand_at_locations = CreateDemandCallback(data)
    demands_callback = demand_at_locations.demand
    vehicle_capacity = 100
    null_capacity_slack = 0
    fix_start_cumul_to_zero = True
    capacity = "Capacity"
    routing.AddDimension(demands_callback,
                         null_capacity_slack,
                         data.vehicle_capacity,
                         fix_start_cumul_to_zero,
                         capacity)

    # Adding a dimension for time-window constraints.
    total_times = CreateTimeCallback(data)
    total_time_callback = total_times.time

    horizon = 24 * 3600
    time = "Time"
    routing.AddDimension(total_time_callback,
                         horizon,
                         horizon,
                         fix_start_cumul_to_zero,
                         time)

    time_dimension = routing.GetDimensionOrDie(time)
    for order in xrange(1, data.num_locations):
        start = data.start_times[order]
        end = data.end_times[order]
        time_dimension.CumulVar(order).SetRange(start, end)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    # Display a solution if any.
    if assignment:
        print_assignment(data, routing, assignment, capacity, time)
    else:
        print('No solution found.')
        sys.exit(2)

if __name__ == '__main__':
    main()
