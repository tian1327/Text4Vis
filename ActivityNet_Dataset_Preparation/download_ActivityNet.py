# download the ActivityNet-200-v1.3 dataset

import fiftyone as fo
import fiftyone.zoo as foz

# Download and load 10 samples from the validation split of ActivityNet 200
dataset = foz.load_zoo_dataset(
    "activitynet-200",
    classes=["Walking the dog"],
    split="validation",
    max_samples=10,
)

session = fo.launch_app(dataset)