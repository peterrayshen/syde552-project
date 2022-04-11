aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 286825838741.dkr.ecr.us-east-1.amazonaws.com
docker pull 286825838741.dkr.ecr.us-east-1.amazonaws.com/snn-train:30796adf1ea3c0cd4a60ae517ed5d5b23aa3fb3b
docker run --name snn-train --gpus all 286825838741.dkr.ecr.us-east-1.amazonaws.com/snn-train:30796adf1ea3c0cd4a60ae517ed5d5b23aa3fb3b
docker cp -a snn-train:/app/results/. ~/train_results/
scp -a -i ~/snn.pem ec2-user@52.207.107.166:/train_results/. /mnt/c/Users/prshe/Documents/github/syde552-project/results
ssh -i ~/snn.pem ec2-user@52.207.107.166