# TDN : Tensor Decomposition for Nodes
This repository contains the Python source codes of two node embedding algorithms: TDNE (Tensor Decomposition-based Node Embedding) and TDNEpS (Tensor Decomposition-based Node Embedding per Slice). We implemented them as extensions of GEM (https://github.com/palash1992/GEM). For details, please read our paper:

S. M. Hamdi, and R. Angryk, "Interpretable Feature Learning of Graphs using Tensor Decomposition," 2019 IEEE 19th International Conference on Data Mining (ICDM), November 8-11, 2019,  Beijing, China.

## Usage
* Git clone GEM (tested on Python 3.6.8, Keras 2.0.2, Tensorflow 1.3.0, Theano 1.0.4, Numpy 1.13.3, Scipy 0.19.1, Networkx 1.11, Scikit-learn 0.19.0). Please see the installation and dependencies of GEM and SNAP (https://github.com/snap-stanford/snap, if you need to run node2vec).
```bash
    git clone https://github.com/palash1992/GEM.git
```
* Git clone TDN (dependency: tensorly 0.4.3)
```bash
    git clone https://github.com/hamdi08/TDN.git
```
* Copy TDNE.py and TDNEpS.py to `./GEM/gem/embedding/`
```bash
    cp ./TDN/TDNE.py ./GEM/gem/embedding/
    cp ./TDN/TDNEpS.py ./GEM/gem/embedding/
```
* Copy Karate.py to `./GEM/`
```bash
    cp ./TDN/Karate.py ./GEM/
```
* Run Karate.py to see network reconstruction performance (MAP and Precision@Np) and 2D visualization of nodes of Karate network.
```bash
    cd GEM/
    python3 Karate.py
```
## Citation
If you use TDNE, please cite any one of the following two papers. If you use TDNEpS, please cite the ICDM paper.
* S. M. Hamdi, S. F. Boubrahimi, and R. Angryk. 2019. Tensor Decomposition-based Node Embedding. In The 28th ACM International Conference on Information and Knowledge Management (CIKM ’19), November 3–7, 2019, Beijing, China. ACM.
* S. M. Hamdi, and R. Angryk, "Interpretable Feature Learning of Graphs using Tensor Decomposition," 2019 IEEE 19th International Conference on Data Mining (ICDM), November 8-11, 2019,  Beijing, China.

