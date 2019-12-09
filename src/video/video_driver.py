
import src.data as data
from src.video.particle_swarm_video import ParticleSwarm
from src.network import Network


def test_particle_swarm():
    car_data = data.get_car_data("../../data/car.data")
    training_data, test_data = car_data.partition(.8)
    network = Network(training_data, test_data, [6, 5, 4], ["acc", "unacc", "good", "vgood"])

    pso = ParticleSwarm(network, pop_size=12, cog_factor=0.5, soc_factor=0.25, inertia=0.5, max_velocity=10000, convergence_size=100)
    pso.train()


def test_particle_swarm_image():
    image_data = data.get_segmentation_data("../../data/segmentation.data")
    training_data, test_data = image_data.partition(.8)
    network = Network(training_data, test_data, [19, 13, 7], ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"])

    pso = ParticleSwarm(network, pop_size=30, cog_factor=0.3, soc_factor=0.2, inertia=0.3, max_velocity=100000, convergence_size=100)
    pso.train()

test_particle_swarm_image()

