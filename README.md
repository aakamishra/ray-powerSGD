# ray-powerSGD
### Ray-based Network Aware PowerSGD

Aakash Mishra, Matt Fu, Chris Zhu

### Initial Setup for our Ray-based network Aware PowerSGD

1) Setup a Venv/Conda environment
2) Install the latest versions of awscli, boto3, numpy, ray, torch, wandb.
3) Add your AWS credentials via the awscli so you can launch the cluster.

### Configure your raypowersgd/cluster.yml
Assuming you want to use 4 nodes, raypowersgd/cluster.yml is mostly defaulted to work with your account. However, be sure to set your --node-ip-address on line 152 to the new headnode that is generated from doing ray up cluster.yml. 

### Step by step instructions to launch an experiment.
1) In your terminal, type ray up cluster.yml
2) Copy the line: ray exec <Your cluster.yml path> 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*
3) In a separate shell, paste it. Once your usage shows 0.0/16.0 CPU (assuming you use our default settings), all nodes have initialized and you can run experiments. 
4) If you want to recreate our experiments exactly, you need to launch each instance in AWS and network bandwidth limit each instance. Use the Linux tc command or the ip link command (should be automically installed for you by us) to limit the bandwidth. 
5) In your original terminal, type ray attach cluster.yml
6) Download vim (sudo apt update -> sudo apt install vim)
7) Enter into the distributed_trianing_test.py file. You will need to change the following:
7a) Change the model near line 507 to what you want to test (resnet.resnet50() or torchvision.models.resnet101())
7b) Change the rank near line 514 to whatever rank you want to use.
7c) Change your WandB key and the project name to yours and what you want, respectively.
8) Launch the experiment yb doing python distributed_training_test.py -n 4 (Note: the number 4 changes if you changed the total number of nodes being spawned off cluster.yml)
9) Now, in WandB as well as in the terminal, you can see detailed information. 
