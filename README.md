# Beyond Poisoning Containment: Defending-Federated-Learning-Against-Adaptive-Multi-Target-Backdoor-Attacks

PACE code
Included methods
PACE method: cumulative per-class exclusion (defense_method: pace).
Baselines: no defense, Multi-Krum, FoolsGold, FLAME, DeepSight, BackdoorIndicator, G²uardFL, and FedDSG.
Oracle control: oracle
Attacks: BadNet, A3FL, and Mirage. Standard benign clients are included.
Running
Install dependencies from requirements.txt, then run from this directory:

python main.py --params path/to/experiment.yaml
