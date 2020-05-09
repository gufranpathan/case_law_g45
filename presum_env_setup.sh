ssh -i ubuntu.pem ec2-user@ec2-13-58-223-11.us-east-2.compute.amazonaws.com
sudo yum update

sudo yum install xfsprogs
sudo mkfs -t xfs /dev/xvdf
sudo mkdir /data
sudo mount /dev/xvdf /data
sudo chmod 777 -R data

sudo yum install git

git clone https://github.com/gufranpathan/case_law_g45.git
sudo yum install python36 python36-pip

python3 -m venv venv
source venv/bin/activate
pip install wheel -y
pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
pip install pytorch-transformers tensorboardX pyrouge gdown jupyter

pyrouge_set_rouge_path ./ROUGE-1.5.5

mkdir model_files && cd model_files && mkdir ext && cd ext
gdown https://drive.google.com/uc?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ
unzip bertext_cnndm_transformer.zip

cd .. && mkdir abs && cd abs
gdown https://drive.google.com/uc?id=1Qksodfu4rHig_-h2UYph8e4XlRN_3rwb
unzip bertsumextabs_cnndm_final_model.zip

cd .. 
mkdir results
mkdir logs

sudo yum install "perl(XML::Parser)" -y
sudo yum install "perl(DB_File)" -y



python train.py -task ext -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path ./data/sample -log_file ./logs/val_abs_bert_cnndm -model_path  ./model_files/ext -sep_optim true -use_interval true -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ./results/ext_bert_cnndm -test_from ./model_files/ext/bertext_cnndm_transformer.pt


#Add train.py
jupyter notebook --generate-config

c = get_config()

# Kernel config
c.IPKernelApp.pylab = 'inline'  # if you want plotting support always in your notebook

# Notebook config
c.NotebookApp.certfile = u'/home/ubuntu/certs/mycert.pem' #location of your certificate file
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False  #so that the ipython notebook does not opens up a browser by default
c.NotebookApp.password = u'sha1:ed206bfd60ce:731f99ffbbdea37f51a55800b39da202e6056dd3'  #the encrypted password we generated above
# Set the port to 8888, the port we set up in the AWS EC2 set-up
c.NotebookApp.port = 8888
