# Adversarial-Attack-JetRacer

This repository explores the implementation of adversarial attacks and defenses on a deep learning model trained on the KITTI dataset. The final goal is to integrate these models into a JetRacer platform to test their robustness in real-world autonomous navigation scenarios.

## Overview

### This project involves:  
1. Training a simple CNN model on the KITTI dataset.
2. Implementing and testing adversarial attack methods, including FGSM and PGD.
3. Developing defense mechanisms to improve model robustness.
4. Preparing for real-world testing by integrating the trained model with JetRacer hardware.  

### Objectives
1. Adversarial Testing: Evaluate the impact of adversarial attacks on model accuracy using the KITTI dataset.  
2. Defense Mechanisms: Implement and test strategies like adversarial training to improve robustness.  
3. JetRacer Integration: Deploy the trained model on JetRacer to analyze real-time performance against adversarial perturbations.  

### Features
1. Attack Implementations: FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent) to generate adversarial examples.
2. Defense Techniques: Includes adversarial training and hyperparameter tuning.
3. Real-World Applicability: Transition from simulation to deployment on the JetRacer platform.  

## Getting Started

### Prerequisites
	• Python 3.8+
	• NVIDIA Jetson Nano or JetRacer hardware (for deployment)
	• PyTorch, torchvision
	• Additional dependencies listed in requirements.txt

### Installation
1. Clone the repository:

		git clone https://github.com/purvavj/Adversarial-Attack-JetRacer.git
		cd Adversarial-Attack-JetRacer

2. Install dependencies:

		pip install -r requirements.txt


3. Ensure the KITTI dataset is downloaded and stored in the data/ folder (excluded from the repository).

## Future Work  
• Enhancing adversarial defense mechanisms.
• Testing additional adversarial attack methods.
• Real-world validation using JetRacer hardware.
