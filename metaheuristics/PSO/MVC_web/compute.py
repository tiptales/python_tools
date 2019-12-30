


import random
import numpy as np


# function that models the problem
def fitness_function(position):
    return position[0] ** 2 + position[1] ** 2 + 1


def extract_s(s):

    if s is not None:
        s_1= s[0]
        s_2 = s[1]

    else:
        s_1 = None
        s_2 = None

    return s_1, s_2


def compute(n_iterations, target_error, n_particles):
    # Some variables to calculate the velocity
    W = 0.5
    c1 = 0.5
    c2 = 0.9
    target = 1
    particle_position_vector = np.array([np.array([(-1) ** (bool(random.getrandbits(1))) * random.random() * 50,
                                               (-1) ** (bool(random.getrandbits(1))) * random.random() * 50]) for _ in
                                     range(n_particles)])
    pbest_position = particle_position_vector
    pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])
    gbest_fitness_value = float('inf')
    gbest_position = np.array([float('inf'), float('inf')])

    velocity_vector = ([np.array([0, 0]) for _ in range(n_particles)])
    iteration = 0
    while iteration < n_iterations:
        for i in range(n_particles):
            fitness_candidate = fitness_function(particle_position_vector[i])
            print(fitness_candidate, ' ', particle_position_vector[i])

            if (pbest_fitness_value[i] > fitness_candidate):
                pbest_fitness_value[i] = fitness_candidate
                pbest_position[i] = particle_position_vector[i]

            if (gbest_fitness_value > fitness_candidate):
                gbest_fitness_value = fitness_candidate
                gbest_position = particle_position_vector[i]

        if (abs(gbest_fitness_value - target) < target_error):
            break

        for i in range(n_particles):
            new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
                    pbest_position[i] - particle_position_vector[i]) + (c2 * random.random()) * (
                                   gbest_position - particle_position_vector[i])
            new_position = new_velocity + particle_position_vector[i]
            particle_position_vector[i] = new_position

        iteration = iteration + 1

        #return tuple([gbest_position, iteration])

        s1, s2 = extract_s(gbest_position)

        #print("The best position is ", gbest_position, "in iteration number ", iteration)
        return s1, s2, iteration


if __name__ == '__main__':


    n_iterations = int(input("Inform the number of iterations: "))
    target_error = float(input("Inform the target error: "))
    n_particles = int(input("Inform the number of particles: "))

    k1, k2, l = compute(n_iterations, target_error, n_particles)

    print(type(k1), type(k2))
    print(type(l))


    print(k1, k2, l)
