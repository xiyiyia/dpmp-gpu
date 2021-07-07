KEY_PEM_DIR=/root/.ssh/id_rsa.pub
KEY_PEM_NAME=id_rsa.pub
PUB_IP_ADDR="$1"
echo "Public address of master node: ${PUB_IP_ADDR}"

ssh -o "StrictHostKeyChecking no" root@${PUB_IP_ADDR} -p 6003
scp -i ${KEY_PEM_DIR} ${KEY_PEM_DIR} root@${PUB_IP_ADDR}:~/.ssh
scp -i ${KEY_PEM_DIR} hosts hosts_address config root@${PUB_IP_ADDR}:~/
scp -i ${KEY_PEM_DIR} -r /home/dpmp/dpmp-gpu/ps_pytorch-master root@${PUB_IP_ADDR}:~/
ssh -i ${KEY_PEM_DIR} root@${PUB_IP_ADDR} 'cp /home/dpmp/dpmp-gpu/ps_pytorch-master/tools/remote_script.sh ~/; bash /home/dpmp/dpmp-gpu/ps_pytorch-master/tools/conda_install.sh'
