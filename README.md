# federated-learning-demo
Imitation learning driving policy network federating on Flower

# Requirements
Python=3.10

Example:
```
conda create -n fldemo python=3.10 -y
conda activate fldemo
pip install -r requirements.txt
```

# Run as a Server
1. Download the initial model weights here:
https://drive.google.com/file/d/1MOJl0HcsvWVgzmGFDNKfa2Dfdr9bBTGD/view?usp=drive_link

1. Run the command below to start the server:
```
python server.py --ckpt_path ./init_weights.ckpt
```

# Run as a Client
1. Place your own dataset under ./data or download our test dataset below. The dataset should consist of RGB images and a csv file of driving parameters generated in CARLA simulator.
https://drive.google.com/file/d/1YVivo12uK7dW-XEyUjslbUND2cQuN5gx/view?usp=drive_link

1. Make sure that the server is running.

1. Run the command below to start a client:
```
python client.py --data_dir ./data
```

# Run a Simulation
You can quickly simulate the federated learning on your environment with:
```
./run.sh
```

