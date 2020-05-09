sudo yum update
sudo yum install xfsprogs
sudo mkdir /data

sudo file -s /dev/nvme1n1
sudo mkfs -t xfs /dev/nvme1n1
sudo mount /dev/nvme1n1 /data
sudo chmod 777 -R data
cd /data


sudo yum install git

git clone https://github.com/gufranpathan/case_law_g45.git
sudo yum install python36 python36-pip

sudo yum -y install gcc gcc++
sudo yum install python36-devel.x86_64

cd case_law_g45
python3 -m venv venv
source venv/bin/activate

pip install wheel setuptools

pip install cytoolz pyrouge gdown
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install transformers
pip install fastNLP rouge

sudo yum install "perl(XML::Parser)" -y
sudo yum install "perl(DB_File)" -y



mkdir matchsumm_data && mkdir matchsumm_models

cd matchsumm_models
gdown https://drive.google.com/uc?id=1jVrQNjVOsLfsbIhw1EbCMD9vqUYERcjI
unzip MatchSum_cnndm_model.zip
mdkir model_zip
mv MatchSum_cnndm_model.zip model_zip/
 
cd ../matchsumm_data
gdown https://drive.google.com/uc?id=1A85SsyVCo1vY3qIOmFRp7-cptKDnZdv8
unzip matchsumm_data.zip

pip install jupyter
jupyter notebook --generate-config
