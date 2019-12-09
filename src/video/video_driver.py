
import src.data as data
from src.video.particle_swarm_video import ParticleSwarm
from src.network import Network


def test_particle_swarm_image():
    image_data = data.get_segmentation_data("../../data/segmentation.data")
    training_data, test_data = image_data.partition(.8)
    network = Network(training_data, test_data, [19, 13, 7], ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"])

    pso = ParticleSwarm(network, pop_size=20, cog_factor=1.0, soc_factor=2.0, inertia=0.05, max_velocity=100000, convergence_size=50)
    pso.train()

    accuracy = network.get_accuracy(test_data)*100
    print("\nConverged!")
    print("\n\nAccuracy on test set: {}%".format(accuracy))


test_particle_swarm_image()


