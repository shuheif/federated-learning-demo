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

2. Run the command below to start the server:
```
python server.py --ckpt ./init_weights.ckpt --address 0.0.0.0:8080
```
3. Once the clients join, it will automatically start the federated training process

# Run as a Client
1. Download our test dataset below. The dataset should consist of RGB images and a CSV file of driving parameters generated in the CARLA simulator.
Small dataset (50 images): https://drive.google.com/file/d/1_OLXeKNu_ueVNLqORjmxGNFVyKqsObUY/view?usp=sharing
Complete daatset (69,350 images): https://drive.google.com/file/d/1YVivo12uK7dW-XEyUjslbUND2cQuN5gx/view?usp=drive_link

2. Make sure that the server is running.

3. Run the command below to start a client:
```
python client.py --server YOUR_SERVER_ADDRESS:8080 --data_dir ./data --client_id YOUR_UNIQUE_ID --seed YOUR_RANDOM_SEED
```

4. Once successfully connecting to the server, it will automatically start the federated learning process

# Run a Simulation
You can also simulate both a server and clients within your single environment by simply running:
```
./run.sh
```

