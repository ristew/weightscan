### weightscan

The goal is to run the hidden layers of a transformer (eg phi-2, stablelm2-1.6) through an autoencoder to get 1024 particles, then visualize the geometry of them with UMAP in 3d. Then, the points data is templated into a three.js visualization that highlights areas of shorter connections and the dynamics over layer-time.

[demo](https://ristew.github.io/weightscan/visualize.html)

controls: space to pause, [] to step, drag/scroll to control camera
