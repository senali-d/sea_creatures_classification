# Sea Creatures Classification

## Problem Statement

To develop a deep learning model using TensorFlow and Keras that can classify images of sea creatures into their respective categories. This involves building a neural network capable of learning complex patterns in image data, achieving high accuracy in predicting the correct category for unseen data.

Problem: Many marine species are endangered due to climate change, overfishing, and habitat destruction.

Marine ecosystems are under significant threat due to climate change, overfishing, and habitat destruction. Automated classification of sea creatures plays a crucial role in addressing these challenges by providing data-driven solutions for conservation efforts.

Solution: Classifying sea creatures can aid in:
1. Identifying critical habitats that need protection.
2. Monitoring the effects of conservation programs.
3. Detecting illegal fishing activities by identifying bycatch or endangered species.

By integrating sea creature classification into conservation workflows, researchers and policymakers can better understand, monitor, and protect marine ecosystems. This technology not only saves time and resources but also provides critical insights into the health of our oceans, helping to safeguard biodiversity for future generations.

### Methods and tools available for sea creature classification

### Steps
1. Collect customer data of Sea animals.
2. Preprocess the data by cleaning, filling missing values, and resampling if necessary.
3. Train the model on the historical data.
5. Validate the model's performance using evaluation metrics.
6. Make classification.

### Dataset

> https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste

dataset has now 23 different Sea animal classes

This dataset contain below categories of images of sea creatures:

1. Seahorse
2. Nudibranchs
3. Sea Urchins
4. Octopus
5. Puffers
6. Rays
7. Whales
8. Eels
9. Crabs
10. Squid
11. Corals
12. Dolphins
13. Seal
14. Penguin
15. Starfish
16. Lobster
17. Jelly Fish
18. Sea Otter
19. Fish
20. Shrimp
21. Clams
22. Fish
23. Turtle Tortoise

Aim is to classify the sea creatures


## How to run this project?

### Prerequisites

- Pipenv, Python virtualenv management tool
- docker and docker-compose

### Project Setup

1. Clone the project from repository

``` 
https://github.com/senali-d/sea_creatures_classification.git
cd sea_creatures_classification
```

2. Build the Docker image from Dockerfile

```
docker build -t sea-creature-model .
```

3. Run the Docker container from the created image

```
docker run -it --rm -p 8080:8080 sea-creature-model:latest
```

4. Test the model

```
pipenv shell
python test.py 
```

If setup is correct, the following output should be displayed:

```
{'Clams': 8.111491203308105,
 'Corals': 0.6925224661827087,
 'Crabs': -9.400032997131348,
 'Dolphin': -10.369836807250977,
 'Eel': -1.9706131219863892,
 'Fish': 1.261847734451294,
 'Jelly Fish': -1.22406804561615,
 'Lobster': -2.665487289428711,
 'Nudibranchs': -3.58951473236084,
 'Octopus': 1.8881003856658936,
 'Otter': -7.621030807495117,
 'Penguin': -4.9838128089904785,
 'Puffers': -0.42829596996307373,
 'Seahorse': -5.163271903991699,
 'Sea Rays': -3.3993330001831055,
 'Sea Urchins': -0.30658140778541565,
 'Seal': -5.755532741546631,
 'Sharks': -1.4640175104141235,
 'Shrimp': 1.9052776098251343,
 'Squid': -5.9878153800964355,
 'Starfish': -4.738603115081787,
 'Turtle_Tortoise': -3.043990135192871,
 'Whale': -3.8058106899261475}
```

## Demo

<img width="1125" alt="Screenshot 2025-01-12 at 3 12 18 AM" src="https://github.com/user-attachments/assets/dacadaeb-51c8-44d9-8f83-e408787b6873" />

<img width="1512" alt="Screenshot 2025-01-12 at 3 49 25 AM" src="https://github.com/user-attachments/assets/89d02601-7838-4c5f-9668-7a8c41b1a12d" />


