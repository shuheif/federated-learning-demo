# federated-learning-demo
Imitation learning driving policy network federating on Flower

- network: ResNet18
- input: an RGB image
- output: a 4-class driving command as a one-hot vector (forward, stop, left, right)

# Requirements
Python=3.10

Example:
```
conda create -n fldemo python=3.10 -y
conda activate fldemo
pip install -r requirements.txt
```

# Dataset
RGB images and driving parameters generated in CARLA simulator

You can download our test dataset here:
https://drive.google.com/file/d/1YVivo12uK7dW-XEyUjslbUND2cQuN5gx/view?usp=drive_link

Once downloaded, make sure to insert the path to your dataset folder in the code as DATA_DIR parameter

# Test Run
You can quickly simulate the federated learning on your environment with:
```
python simulation.py
```

