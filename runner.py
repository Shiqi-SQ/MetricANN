import os

# Create directories
os.makedirs('distributed', exist_ok=True)
os.makedirs('utils', exist_ok=True)

# distributed/client.py
with open('distributed/client.py', 'w') as f:
    f.write("""class DistributedClient:
    def __init__(self, config):
        pass

    def send_query(self, embedding):
        \"\"\"Send embedding to distributed index service and return results\"\"\"
        pass

    def add_node(self, node_config):
        \"\"\"Add a new index/compute node to the distributed cluster\"\"\"
        pass
""")

# utils/logging.py
with open('utils/logging.py', 'w') as f:
    f.write("""import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
""")

# utils/metrics.py
with open('utils/metrics.py', 'w') as f:
    f.write("""import numpy as np

def top_k_accuracy(true_labels, predicted_lists, k):
    correct = sum(1 for tl, preds in zip(true_labels, predicted_lists) if tl in preds[:k])
    return correct / len(true_labels)

def mean_reciprocal_rank(true_labels, predicted_lists):
    rr_values = []
    for tl, preds in zip(true_labels, predicted_lists):
        try:
            rank = preds.index(tl) + 1
            rr_values.append(1.0 / rank)
        except ValueError:
            rr_values.append(0.0)
    return np.mean(rr_values)
""")

# requirements.txt
with open('requirements.txt', 'w') as f:
    f.write("""torch>=1.13.0
torchvision
faiss-cpu
hnswlib
Pillow
numpy
""")

print("Distributed client and utils files created, requirements.txt generated.")
