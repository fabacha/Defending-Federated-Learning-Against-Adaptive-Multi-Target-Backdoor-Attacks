# Beyond Poisoning Containment: Defending-Federated-Learning-Against-Adaptive-Multi-Target-Backdoor-Attacks


Included methods
PACE method: cumulative per-class exclusion (defense_method: pace).
Baselines: no defense, Multi-Krum, FoolsGold, FLAME, DeepSight, BackdoorIndicator, G²uardFL, and FedDSG.
Oracle control: oracle
Attacks: BadNet, A3FL, and Mirage. Standard benign clients are included.
Running
Install dependencies from requirements.txt, then run from this directory:

To run python main.py --params path/to/experiment.yaml

PACE CIFAR10 : python main.py --params ./yamls/CIFAR10/Mirage_PACE.yaml

PACE CIFAR100: python main.py --params ./yamls/CIFAR100/Mirage_PACE.yaml --dataset CIFAR100

Reference: https://github.com/NUAA-SmartSensing/Mirage/tree/main/Mirage

