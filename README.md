# DN3 - Deep Neural Networks for Neuro-physiology
Bridge for training deep neural-network models with neuroscientific data managed using MNE.

Focused on:
 * Minimizing boilerplate for DNN powered BCI classifiers and processors 
 * Rapid integration and extension to _new_ datasets by providing a yaml interface to dataset construction
 * Platform for accessing state-of-the-art
   * Architectures (potentially with pretrained weights) 
   * Pre-processing and data transformations
   
See guides and documentation at:
https://dn3.readthedocs.io/en/latest/

Associated pre-print *(article under review)* can be found at:
 https://www.biorxiv.org/content/10.1101/2020.12.17.423197v1
 
*Please consider citing the above in any scholarly work that uses this library.*

## Requirements:
 * python >= 3.5
 * pytorch >= 1.3
 * mne >= 0.20
 * pyyaml
 * pyyaml-include
 * numpy
 * pandas
 * tqdm
 
