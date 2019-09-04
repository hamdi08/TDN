# TDN : Tensor Decomposition-based Node representation learning
This repository contains the Python source codes of two node embedding algorithms: TDNE (Tensor Decomposition-based Node Embedding) and TDNEpS (Tensor Decomposition-based Node Embedding per Slice). We implemented them as extensions of GEM (https://github.com/palash1992/GEM). For details, please read our paper:

S. M. Hamdi, and R. Angryk, "Interpretable Feature Learning of Graphs using Tensor Decomposition," 2019 IEEE 19th International Conference on Data Mining (ICDM), November 8-11, 2019,  Beijing, China.

## Usage
* Git clone GEM
```bash
    git clone https://github.com/palash1992/GEM.git
```
* Git clone TDN
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
