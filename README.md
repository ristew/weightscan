### weightscan

First, run the hidden layers of a transformer language model (by default stablelm2-1.6) through an autoencoder to get a fixed amount of 16d particles (768). These vectors are then passed to UMAP to be projected into 3d points. The points data is templated into a three.js visualization that highlights areas of shorter connections and the dynamics over layer-time with a few keyboard controls. 

[demo](https://ristew.github.io/weightscan/visualize.html) (may take a bit to load)

Demo prompt:

    Pattern matching

    input:
    0, 0
    0, 1
    output:
    1, 0
    0, 0
    input:
    0, 0
    0, 9
    output:
    3, 0
    0, 0
    input:
    0, 0
    0, 4
    output:

controls: space to pause, [] to step, drag/scroll to control camera
