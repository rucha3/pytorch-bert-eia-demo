```python
!pip show torch
```

    Name: torch
    Version: 1.3.1
    Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
    Home-page: https://pytorch.org/
    Author: PyTorch Team
    Author-email: packages@pytorch.org
    License: BSD-3
    Location: /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages
    Requires: numpy
    Required-by: torchvision, fastai



```python
!pip install 'sagemaker[local]' --upgrade
```

    Collecting sagemaker[local]
      Downloading sagemaker-2.20.0.tar.gz (401 kB)
    [K     |████████████████████████████████| 401 kB 37.9 MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: attrs in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (19.3.0)
    Collecting boto3>=1.16.32
      Downloading boto3-1.16.38-py2.py3-none-any.whl (130 kB)
    [K     |████████████████████████████████| 130 kB 91.3 MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: google-pasta in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (0.2.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.9.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (1.18.1)
    Requirement already satisfied, skipping upgrade: protobuf>=3.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (3.11.4)
    Requirement already satisfied, skipping upgrade: protobuf3-to-dict>=0.1.5 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (0.1.5)
    Collecting smdebug_rulesconfig>=1.0.0
      Downloading smdebug_rulesconfig-1.0.0-py3-none-any.whl (14 kB)
    Requirement already satisfied, skipping upgrade: importlib-metadata>=1.4.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (2.0.0)
    Requirement already satisfied, skipping upgrade: packaging>=20.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (20.1)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (1.25.10)
    Requirement already satisfied, skipping upgrade: docker-compose>=1.25.2 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (1.27.4)
    Requirement already satisfied, skipping upgrade: PyYAML<6,>=5.3 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sagemaker[local]) (5.3.1)
    Collecting botocore<1.20.0,>=1.19.38
      Downloading botocore-1.19.38-py2.py3-none-any.whl (7.1 MB)
    [K     |████████████████████████████████| 7.1 MB 81.8 MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: s3transfer<0.4.0,>=0.3.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from boto3>=1.16.32->sagemaker[local]) (0.3.3)
    Requirement already satisfied, skipping upgrade: jmespath<1.0.0,>=0.7.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from boto3>=1.16.32->sagemaker[local]) (0.10.0)
    Requirement already satisfied, skipping upgrade: six in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from google-pasta->sagemaker[local]) (1.14.0)
    Requirement already satisfied, skipping upgrade: setuptools in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from protobuf>=3.1->sagemaker[local]) (45.2.0.post20200210)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from importlib-metadata>=1.4.0->sagemaker[local]) (2.2.0)
    Requirement already satisfied, skipping upgrade: pyparsing>=2.0.2 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from packaging>=20.0->sagemaker[local]) (2.4.6)
    Requirement already satisfied, skipping upgrade: docker[ssh]<5,>=4.3.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (4.3.1)
    Requirement already satisfied, skipping upgrade: docopt<1,>=0.6.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (0.6.2)
    Requirement already satisfied, skipping upgrade: python-dotenv<1,>=0.13.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (0.15.0)
    Requirement already satisfied, skipping upgrade: jsonschema<4,>=2.5.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (3.2.0)
    Requirement already satisfied, skipping upgrade: dockerpty<1,>=0.4.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (0.4.1)
    Requirement already satisfied, skipping upgrade: texttable<2,>=0.9.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (1.6.3)
    Requirement already satisfied, skipping upgrade: websocket-client<1,>=0.32.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (0.57.0)
    Requirement already satisfied, skipping upgrade: distro<2,>=1.5.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (1.5.0)
    Requirement already satisfied, skipping upgrade: requests<3,>=2.20.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (2.22.0)
    Requirement already satisfied, skipping upgrade: cached-property<2,>=1.2.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker-compose>=1.25.2->sagemaker[local]) (1.5.2)
    Requirement already satisfied, skipping upgrade: python-dateutil<3.0.0,>=2.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from botocore<1.20.0,>=1.19.38->boto3>=1.16.32->sagemaker[local]) (2.8.1)
    Requirement already satisfied, skipping upgrade: paramiko>=2.4.2; extra == "ssh" in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from docker[ssh]<5,>=4.3.1->docker-compose>=1.25.2->sagemaker[local]) (2.7.2)
    Requirement already satisfied, skipping upgrade: pyrsistent>=0.14.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from jsonschema<4,>=2.5.1->docker-compose>=1.25.2->sagemaker[local]) (0.15.7)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from requests<3,>=2.20.0->docker-compose>=1.25.2->sagemaker[local]) (2020.6.20)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from requests<3,>=2.20.0->docker-compose>=1.25.2->sagemaker[local]) (2.8)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from requests<3,>=2.20.0->docker-compose>=1.25.2->sagemaker[local]) (3.0.4)
    Requirement already satisfied, skipping upgrade: cryptography>=2.5 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from paramiko>=2.4.2; extra == "ssh"->docker[ssh]<5,>=4.3.1->docker-compose>=1.25.2->sagemaker[local]) (2.8)
    Requirement already satisfied, skipping upgrade: pynacl>=1.0.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from paramiko>=2.4.2; extra == "ssh"->docker[ssh]<5,>=4.3.1->docker-compose>=1.25.2->sagemaker[local]) (1.4.0)
    Requirement already satisfied, skipping upgrade: bcrypt>=3.1.3 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from paramiko>=2.4.2; extra == "ssh"->docker[ssh]<5,>=4.3.1->docker-compose>=1.25.2->sagemaker[local]) (3.2.0)
    Requirement already satisfied, skipping upgrade: cffi!=1.11.3,>=1.8 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from cryptography>=2.5->paramiko>=2.4.2; extra == "ssh"->docker[ssh]<5,>=4.3.1->docker-compose>=1.25.2->sagemaker[local]) (1.14.0)
    Requirement already satisfied, skipping upgrade: pycparser in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from cffi!=1.11.3,>=1.8->cryptography>=2.5->paramiko>=2.4.2; extra == "ssh"->docker[ssh]<5,>=4.3.1->docker-compose>=1.25.2->sagemaker[local]) (2.19)
    Building wheels for collected packages: sagemaker
      Building wheel for sagemaker (setup.py) ... [?25ldone
    [?25h  Created wheel for sagemaker: filename=sagemaker-2.20.0-py2.py3-none-any.whl size=557136 sha256=e224fd12ba4dfe9463999b0276dc7dab1e8aa7f217fede64438eb8767c0b0bce
      Stored in directory: /home/ec2-user/.cache/pip/wheels/18/82/d0/58c578a8b96322ca48dca9d5799d8f4d094face066687174b9
    Successfully built sagemaker
    [31mERROR: awscli 1.18.179 has requirement botocore==1.19.19, but you'll have botocore 1.19.38 which is incompatible.[0m
    Installing collected packages: botocore, boto3, smdebug-rulesconfig, sagemaker
      Attempting uninstall: botocore
        Found existing installation: botocore 1.19.19
        Uninstalling botocore-1.19.19:
          Successfully uninstalled botocore-1.19.19
      Attempting uninstall: boto3
        Found existing installation: boto3 1.16.19
        Uninstalling boto3-1.16.19:
          Successfully uninstalled boto3-1.16.19
      Attempting uninstall: smdebug-rulesconfig
        Found existing installation: smdebug-rulesconfig 0.1.6
        Uninstalling smdebug-rulesconfig-0.1.6:
          Successfully uninstalled smdebug-rulesconfig-0.1.6
      Attempting uninstall: sagemaker
        Found existing installation: sagemaker 2.16.4.dev0
        Uninstalling sagemaker-2.16.4.dev0:
          Successfully uninstalled sagemaker-2.16.4.dev0
    Successfully installed boto3-1.16.38 botocore-1.19.38 sagemaker-2.20.0 smdebug-rulesconfig-1.0.0
    [33mWARNING: You are using pip version 20.0.2; however, version 20.3.3 is available.
    You should consider upgrading via the '/home/ec2-user/anaconda3/envs/pytorch_p36/bin/python -m pip install --upgrade pip' command.[0m



```python
import os
import numpy as np
import pandas as pd
import sagemaker
from sagemaker.local import LocalSession

# sagemaker_session = sagemaker.Session()
sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

bucket = "indix-users"
prefix = "rucha/sagemaker/DEMO-pytorch-bert"

role = sagemaker.get_execution_role()
```


```python
# Download data

if not os.path.exists("./cola_public_1.1.zip"):
    !curl -o ./cola_public_1.1.zip https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
if not os.path.exists("./cola_public/"):
    !unzip cola_public_1.1.zip
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  249k  100  249k    0     0  2899k      0 --:--:-- --:--:-- --:--:-- 2899k
    Archive:  cola_public_1.1.zip
       creating: cola_public/
      inflating: cola_public/README      
       creating: cola_public/tokenized/
      inflating: cola_public/tokenized/in_domain_dev.tsv  
      inflating: cola_public/tokenized/in_domain_train.tsv  
      inflating: cola_public/tokenized/out_of_domain_dev.tsv  
       creating: cola_public/raw/
      inflating: cola_public/raw/in_domain_dev.tsv  
      inflating: cola_public/raw/in_domain_train.tsv  
      inflating: cola_public/raw/out_of_domain_dev.tsv  



```python
# Get sentences and labels
# Let us take a quick look at our data. First we read in the training data. The only two columns we need are the sentence itself and its label.

df = pd.read_csv(
    "./cola_public/raw/in_domain_train.tsv",
    sep="\t",
    header=None,
    usecols=[1, 3],
    names=["label", "sentence"],
)
sentences = df.sentence.values
labels = df.label.values
```


```python
len(sentences)
```




    8551




```python
print(sentences[20:25])
print(labels[20:25])
```

    ['The professor talked us.' 'We yelled ourselves hoarse.'
     'We yelled ourselves.' 'We yelled Harry hoarse.'
     'Harry coughed himself into a fit.']
    [0 1 0 0 1]



```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(df)
train.to_csv("./cola_public/train.csv", index=False)
test.to_csv("./cola_public/test.csv", index=False)
```


```python
inputs_train = sagemaker_session.upload_data("./cola_public/train.csv", bucket=bucket, key_prefix=prefix)
inputs_test = sagemaker_session.upload_data("./cola_public/test.csv", bucket=bucket, key_prefix=prefix)
```


```python
# Run training
# Training script

!pygmentize code/train_deploy.py
```

    [34mimport[39;49;00m [04m[36margparse[39;49;00m
    [34mimport[39;49;00m [04m[36mjson[39;49;00m
    [34mimport[39;49;00m [04m[36mlogging[39;49;00m
    [34mimport[39;49;00m [04m[36mos[39;49;00m
    [34mimport[39;49;00m [04m[36msys[39;49;00m
    
    [34mimport[39;49;00m [04m[36mnumpy[39;49;00m [34mas[39;49;00m [04m[36mnp[39;49;00m
    [34mimport[39;49;00m [04m[36mpandas[39;49;00m [34mas[39;49;00m [04m[36mpd[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36mdistributed[39;49;00m [34mas[39;49;00m [04m[36mdist[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36mutils[39;49;00m[04m[36m.[39;49;00m[04m[36mdata[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36mutils[39;49;00m[04m[36m.[39;49;00m[04m[36mdata[39;49;00m[04m[36m.[39;49;00m[04m[36mdistributed[39;49;00m
    [34mfrom[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36mutils[39;49;00m[04m[36m.[39;49;00m[04m[36mdata[39;49;00m [34mimport[39;49;00m DataLoader, RandomSampler, TensorDataset
    [34mfrom[39;49;00m [04m[36mtransformers[39;49;00m [34mimport[39;49;00m AdamW, BertForSequenceClassification, BertTokenizer
    
    logger = logging.getLogger([31m__name__[39;49;00m)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    MAX_LEN = [34m64[39;49;00m  [37m# this is the max length of the sentence[39;49;00m
    
    [36mprint[39;49;00m([33m"[39;49;00m[33mLoading BERT tokenizer...[39;49;00m[33m"[39;49;00m)
    tokenizer = BertTokenizer.from_pretrained([33m"[39;49;00m[33mbert-base-uncased[39;49;00m[33m"[39;49;00m, do_lower_case=[34mTrue[39;49;00m)
    
    
    [34mdef[39;49;00m [32mflat_accuracy[39;49;00m(preds, labels):
        pred_flat = np.argmax(preds, axis=[34m1[39;49;00m).flatten()
        labels_flat = labels.flatten()
        [34mreturn[39;49;00m np.sum(pred_flat == labels_flat) / [36mlen[39;49;00m(labels_flat)
    
    
    [34mdef[39;49;00m [32m_get_train_data_loader[39;49;00m(batch_size, training_dir, is_distributed):
        logger.info([33m"[39;49;00m[33mGet train data loader[39;49;00m[33m"[39;49;00m)
    
        dataset = pd.read_csv(os.path.join(training_dir, [33m"[39;49;00m[33mtrain.csv[39;49;00m[33m"[39;49;00m))
        sentences = dataset.sentence.values
        labels = dataset.label.values
    
        input_ids = []
        [34mfor[39;49;00m sent [35min[39;49;00m sentences:
            encoded_sent = tokenizer.encode(sent, add_special_tokens=[34mTrue[39;49;00m)
            input_ids.append(encoded_sent)
    
        [37m# pad shorter sentences[39;49;00m
        input_ids_padded = []
        [34mfor[39;49;00m i [35min[39;49;00m input_ids:
            [34mwhile[39;49;00m [36mlen[39;49;00m(i) < MAX_LEN:
                i.append([34m0[39;49;00m)
            input_ids_padded.append(i)
        input_ids = input_ids_padded
    
        [37m# mask; 0: added, 1: otherwise[39;49;00m
        attention_masks = []
        [37m# For each sentence...[39;49;00m
        [34mfor[39;49;00m sent [35min[39;49;00m input_ids:
            att_mask = [[36mint[39;49;00m(token_id > [34m0[39;49;00m) [34mfor[39;49;00m token_id [35min[39;49;00m sent]
            attention_masks.append(att_mask)
    
        [37m# convert to PyTorch data types.[39;49;00m
        train_inputs = torch.tensor(input_ids)
        train_labels = torch.tensor(labels)
        train_masks = torch.tensor(attention_masks)
    
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        [34mif[39;49;00m is_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        [34melse[39;49;00m:
            train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
        [34mreturn[39;49;00m train_dataloader
    
    
    [34mdef[39;49;00m [32m_get_test_data_loader[39;49;00m(test_batch_size, training_dir):
        dataset = pd.read_csv(os.path.join(training_dir, [33m"[39;49;00m[33mtest.csv[39;49;00m[33m"[39;49;00m))
        sentences = dataset.sentence.values
        labels = dataset.label.values
    
        input_ids = []
        [34mfor[39;49;00m sent [35min[39;49;00m sentences:
            encoded_sent = tokenizer.encode(sent, add_special_tokens=[34mTrue[39;49;00m)
            input_ids.append(encoded_sent)
    
        [37m# pad shorter sentences[39;49;00m
        input_ids_padded = []
        [34mfor[39;49;00m i [35min[39;49;00m input_ids:
            [34mwhile[39;49;00m [36mlen[39;49;00m(i) < MAX_LEN:
                i.append([34m0[39;49;00m)
            input_ids_padded.append(i)
        input_ids = input_ids_padded
    
        [37m# mask; 0: added, 1: otherwise[39;49;00m
        attention_masks = []
        [37m# For each sentence...[39;49;00m
        [34mfor[39;49;00m sent [35min[39;49;00m input_ids:
            att_mask = [[36mint[39;49;00m(token_id > [34m0[39;49;00m) [34mfor[39;49;00m token_id [35min[39;49;00m sent]
            attention_masks.append(att_mask)
    
        [37m# convert to PyTorch data types.[39;49;00m
        train_inputs = torch.tensor(input_ids)
        train_labels = torch.tensor(labels)
        train_masks = torch.tensor(attention_masks)
    
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=test_batch_size)
    
        [34mreturn[39;49;00m train_dataloader
    
    
    [34mdef[39;49;00m [32mtrain[39;49;00m(args):
        is_distributed = [36mlen[39;49;00m(args.hosts) > [34m1[39;49;00m [35mand[39;49;00m args.backend [35mis[39;49;00m [35mnot[39;49;00m [34mNone[39;49;00m
        logger.debug([33m"[39;49;00m[33mDistributed training - [39;49;00m[33m%s[39;49;00m[33m"[39;49;00m, is_distributed)
        use_cuda = args.num_gpus > [34m0[39;49;00m
        logger.debug([33m"[39;49;00m[33mNumber of gpus available - [39;49;00m[33m%d[39;49;00m[33m"[39;49;00m, args.num_gpus)
        device = torch.device([33m"[39;49;00m[33mcuda[39;49;00m[33m"[39;49;00m [34mif[39;49;00m use_cuda [34melse[39;49;00m [33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m)
    
        [34mif[39;49;00m is_distributed:
            [37m# Initialize the distributed environment.[39;49;00m
            world_size = [36mlen[39;49;00m(args.hosts)
            os.environ[[33m"[39;49;00m[33mWORLD_SIZE[39;49;00m[33m"[39;49;00m] = [36mstr[39;49;00m(world_size)
            host_rank = args.hosts.index(args.current_host)
            os.environ[[33m"[39;49;00m[33mRANK[39;49;00m[33m"[39;49;00m] = [36mstr[39;49;00m(host_rank)
            dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
            logger.info(
                [33m"[39;49;00m[33mInitialized the distributed environment: [39;49;00m[33m'[39;49;00m[33m%s[39;49;00m[33m'[39;49;00m[33m backend on [39;49;00m[33m%d[39;49;00m[33m nodes. [39;49;00m[33m"[39;49;00m
                [33m"[39;49;00m[33mCurrent host rank is [39;49;00m[33m%d[39;49;00m[33m. Number of gpus: [39;49;00m[33m%d[39;49;00m[33m"[39;49;00m,
                args.backend, dist.get_world_size(),
                dist.get_rank(), args.num_gpus
            )
    
        [37m# set the seed for generating random numbers[39;49;00m
        torch.manual_seed(args.seed)
        [34mif[39;49;00m use_cuda:
            torch.cuda.manual_seed(args.seed)
    
        train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed)
        test_loader = _get_test_data_loader(args.test_batch_size, args.test)
    
        logger.debug(
            [33m"[39;49;00m[33mProcesses [39;49;00m[33m{}[39;49;00m[33m/[39;49;00m[33m{}[39;49;00m[33m ([39;49;00m[33m{:.0f}[39;49;00m[33m%[39;49;00m[33m) of train data[39;49;00m[33m"[39;49;00m.format(
                [36mlen[39;49;00m(train_loader.sampler),
                [36mlen[39;49;00m(train_loader.dataset),
                [34m100.0[39;49;00m * [36mlen[39;49;00m(train_loader.sampler) / [36mlen[39;49;00m(train_loader.dataset),
            )
        )
    
        logger.debug(
            [33m"[39;49;00m[33mProcesses [39;49;00m[33m{}[39;49;00m[33m/[39;49;00m[33m{}[39;49;00m[33m ([39;49;00m[33m{:.0f}[39;49;00m[33m%[39;49;00m[33m) of test data[39;49;00m[33m"[39;49;00m.format(
                [36mlen[39;49;00m(test_loader.sampler),
                [36mlen[39;49;00m(test_loader.dataset),
                [34m100.0[39;49;00m * [36mlen[39;49;00m(test_loader.sampler) / [36mlen[39;49;00m(test_loader.dataset),
            )
        )
    
        logger.info([33m"[39;49;00m[33mStarting BertForSequenceClassification[39;49;00m[33m\n[39;49;00m[33m"[39;49;00m)
        model = BertForSequenceClassification.from_pretrained(
            [33m"[39;49;00m[33mbert-base-uncased[39;49;00m[33m"[39;49;00m,  [37m# Use the 12-layer BERT model, with an uncased vocab.[39;49;00m
            num_labels=args.num_labels,  [37m# The number of output labels--2 for binary classification.[39;49;00m
            output_attentions=[34mFalse[39;49;00m,  [37m# Whether the model returns attentions weights.[39;49;00m
            output_hidden_states=[34mFalse[39;49;00m,  [37m# Whether the model returns all hidden-states.[39;49;00m
        )
    
        model = model.to(device)
        [34mif[39;49;00m is_distributed [35mand[39;49;00m use_cuda:
            [37m# multi-machine multi-gpu case[39;49;00m
            model = torch.nn.parallel.DistributedDataParallel(model)
        [34melse[39;49;00m:
            [37m# single-machine multi-gpu case or single-machine or multi-machine cpu case[39;49;00m
            model = torch.nn.DataParallel(model)
        optimizer = AdamW(
            model.parameters(),
            lr=[34m2e-5[39;49;00m,  [37m# args.learning_rate - default is 5e-5, our notebook had 2e-5[39;49;00m
            eps=[34m1e-8[39;49;00m,  [37m# args.adam_epsilon - default is 1e-8.[39;49;00m
        )
    
        logger.info([33m"[39;49;00m[33mEnd of defining BertForSequenceClassification[39;49;00m[33m\n[39;49;00m[33m"[39;49;00m)
        [34mfor[39;49;00m epoch [35min[39;49;00m [36mrange[39;49;00m([34m1[39;49;00m, args.epochs + [34m1[39;49;00m):
            total_loss = [34m0[39;49;00m
            model.train()
            [34mfor[39;49;00m step, batch [35min[39;49;00m [36menumerate[39;49;00m(train_loader):
                b_input_ids = batch[[34m0[39;49;00m].to(device)
                b_input_mask = batch[[34m1[39;49;00m].to(device)
                b_labels = batch[[34m2[39;49;00m].to(device)
                model.zero_grad()
    
                outputs = model(b_input_ids, token_type_ids=[34mNone[39;49;00m, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[[34m0[39;49;00m]
    
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), [34m1.0[39;49;00m)
                [37m# modified based on their gradients, the learning rate, etc.[39;49;00m
                optimizer.step()
                [34mif[39;49;00m step % args.log_interval == [34m0[39;49;00m:
                    logger.info(
                        [33m"[39;49;00m[33mTrain Epoch: [39;49;00m[33m{}[39;49;00m[33m [[39;49;00m[33m{}[39;49;00m[33m/[39;49;00m[33m{}[39;49;00m[33m ([39;49;00m[33m{:.0f}[39;49;00m[33m%[39;49;00m[33m)] Loss: [39;49;00m[33m{:.6f}[39;49;00m[33m"[39;49;00m.format(
                            epoch,
                            step * [36mlen[39;49;00m(batch[[34m0[39;49;00m]),
                            [36mlen[39;49;00m(train_loader.sampler),
                            [34m100.0[39;49;00m * step / [36mlen[39;49;00m(train_loader),
                            loss.item(),
                        )
                    )
    
            logger.info([33m"[39;49;00m[33mAverage training loss: [39;49;00m[33m%f[39;49;00m[33m\n[39;49;00m[33m"[39;49;00m, total_loss / [36mlen[39;49;00m(train_loader))
    
            test(model, test_loader, device)
    
        logger.info([33m"[39;49;00m[33mSaving tuned model.[39;49;00m[33m"[39;49;00m)
        model_2_save = model.module [34mif[39;49;00m [36mhasattr[39;49;00m(model, [33m"[39;49;00m[33mmodule[39;49;00m[33m"[39;49;00m) [34melse[39;49;00m model
        model_2_save.save_pretrained(save_directory=args.model_dir)
    
    
    [34mdef[39;49;00m [32mtest[39;49;00m(model, test_loader, device):
        model.eval()
        _, eval_accuracy = [34m0[39;49;00m, [34m0[39;49;00m
    
        [34mwith[39;49;00m torch.no_grad():
            [34mfor[39;49;00m batch [35min[39;49;00m test_loader:
                b_input_ids = batch[[34m0[39;49;00m].to(device)
                b_input_mask = batch[[34m1[39;49;00m].to(device)
                b_labels = batch[[34m2[39;49;00m].to(device)
    
                outputs = model(b_input_ids, token_type_ids=[34mNone[39;49;00m, attention_mask=b_input_mask)
                logits = outputs[[34m0[39;49;00m]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to([33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m).numpy()
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
    
        logger.info([33m"[39;49;00m[33mTest set: Accuracy: [39;49;00m[33m%f[39;49;00m[33m\n[39;49;00m[33m"[39;49;00m, tmp_eval_accuracy)
    
    
    [34mdef[39;49;00m [32mmodel_fn[39;49;00m(model_dir):
        device = torch.device([33m"[39;49;00m[33mcuda[39;49;00m[33m"[39;49;00m [34mif[39;49;00m torch.cuda.is_available() [34melse[39;49;00m [33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m)
    
        model = BertForSequenceClassification.from_pretrained(model_dir)
        [34mreturn[39;49;00m model.to(device)
    
    
    [34mdef[39;49;00m [32minput_fn[39;49;00m(request_body, request_content_type):
        [33m"""An input_fn that loads a pickled tensor"""[39;49;00m
        [34mif[39;49;00m request_content_type == [33m"[39;49;00m[33mapplication/json[39;49;00m[33m"[39;49;00m:
            sentence = json.loads(request_body)
    
            input_ids = []
            encoded_sent = tokenizer.encode(sentence, add_special_tokens=[34mTrue[39;49;00m)
            input_ids.append(encoded_sent)
    
            [37m# pad shorter sentences[39;49;00m
            input_ids_padded = []
            [34mfor[39;49;00m i [35min[39;49;00m input_ids:
                [34mwhile[39;49;00m [36mlen[39;49;00m(i) < MAX_LEN:
                    i.append([34m0[39;49;00m)
                input_ids_padded.append(i)
            input_ids = input_ids_padded
    
            [37m# mask; 0: added, 1: otherwise[39;49;00m
            attention_masks = []
            [37m# For each sentence...[39;49;00m
            [34mfor[39;49;00m sent [35min[39;49;00m input_ids:
                att_mask = [[36mint[39;49;00m(token_id > [34m0[39;49;00m) [34mfor[39;49;00m token_id [35min[39;49;00m sent]
                attention_masks.append(att_mask)
    
            [37m# convert to PyTorch data types.[39;49;00m
            train_inputs = torch.tensor(input_ids)
            train_masks = torch.tensor(attention_masks)
    
            [34mreturn[39;49;00m train_inputs, train_masks
    
        [34mraise[39;49;00m [36mValueError[39;49;00m([33m"[39;49;00m[33mUnsupported content type: [39;49;00m[33m{}[39;49;00m[33m"[39;49;00m.format(request_content_type))
    
    
    [34mdef[39;49;00m [32mpredict_fn[39;49;00m(input_data, model):
        device = torch.device([33m"[39;49;00m[33mcuda[39;49;00m[33m"[39;49;00m [34mif[39;49;00m torch.cuda.is_available() [34melse[39;49;00m [33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m)
        model.to(device)
        model.eval()
    
        input_id, input_mask = input_data
        input_id = input_id.to(device)
        input_mask = input_mask.to(device)
        [34mwith[39;49;00m torch.no_grad():
            [34mreturn[39;49;00m model(input_id, token_type_ids=[34mNone[39;49;00m, attention_mask=input_mask)[[34m0[39;49;00m]
    
    
    [34mif[39;49;00m [31m__name__[39;49;00m == [33m"[39;49;00m[33m__main__[39;49;00m[33m"[39;49;00m:
        parser = argparse.ArgumentParser()
    
        [37m# Data and model checkpoints directories[39;49;00m
        parser.add_argument(
            [33m"[39;49;00m[33m--num_labels[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=[34m2[39;49;00m, metavar=[33m"[39;49;00m[33mN[39;49;00m[33m"[39;49;00m, help=[33m"[39;49;00m[33minput batch size for training (default: 64)[39;49;00m[33m"[39;49;00m
        )
    
        parser.add_argument(
            [33m"[39;49;00m[33m--batch-size[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=[34m64[39;49;00m, metavar=[33m"[39;49;00m[33mN[39;49;00m[33m"[39;49;00m, help=[33m"[39;49;00m[33minput batch size for training (default: 64)[39;49;00m[33m"[39;49;00m
        )
        parser.add_argument(
            [33m"[39;49;00m[33m--test-batch-size[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=[34m1000[39;49;00m, metavar=[33m"[39;49;00m[33mN[39;49;00m[33m"[39;49;00m, help=[33m"[39;49;00m[33minput batch size for testing (default: 1000)[39;49;00m[33m"[39;49;00m
        )
        parser.add_argument([33m"[39;49;00m[33m--epochs[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=[34m2[39;49;00m, metavar=[33m"[39;49;00m[33mN[39;49;00m[33m"[39;49;00m, help=[33m"[39;49;00m[33mnumber of epochs to train (default: 10)[39;49;00m[33m"[39;49;00m)
        parser.add_argument([33m"[39;49;00m[33m--lr[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mfloat[39;49;00m, default=[34m0.01[39;49;00m, metavar=[33m"[39;49;00m[33mLR[39;49;00m[33m"[39;49;00m, help=[33m"[39;49;00m[33mlearning rate (default: 0.01)[39;49;00m[33m"[39;49;00m)
        parser.add_argument([33m"[39;49;00m[33m--momentum[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mfloat[39;49;00m, default=[34m0.5[39;49;00m, metavar=[33m"[39;49;00m[33mM[39;49;00m[33m"[39;49;00m, help=[33m"[39;49;00m[33mSGD momentum (default: 0.5)[39;49;00m[33m"[39;49;00m)
        parser.add_argument([33m"[39;49;00m[33m--seed[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=[34m1[39;49;00m, metavar=[33m"[39;49;00m[33mS[39;49;00m[33m"[39;49;00m, help=[33m"[39;49;00m[33mrandom seed (default: 1)[39;49;00m[33m"[39;49;00m)
        parser.add_argument(
            [33m"[39;49;00m[33m--log-interval[39;49;00m[33m"[39;49;00m,
            [36mtype[39;49;00m=[36mint[39;49;00m,
            default=[34m50[39;49;00m,
            metavar=[33m"[39;49;00m[33mN[39;49;00m[33m"[39;49;00m,
            help=[33m"[39;49;00m[33mhow many batches to wait before logging training status[39;49;00m[33m"[39;49;00m,
        )
        parser.add_argument(
            [33m"[39;49;00m[33m--backend[39;49;00m[33m"[39;49;00m,
            [36mtype[39;49;00m=[36mstr[39;49;00m,
            default=[34mNone[39;49;00m,
            help=[33m"[39;49;00m[33mbackend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)[39;49;00m[33m"[39;49;00m,
        )
    
        [37m# Container environment[39;49;00m
        parser.add_argument([33m"[39;49;00m[33m--hosts[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mlist[39;49;00m, default=json.loads(os.environ[[33m"[39;49;00m[33mSM_HOSTS[39;49;00m[33m"[39;49;00m]))
        parser.add_argument([33m"[39;49;00m[33m--current-host[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_CURRENT_HOST[39;49;00m[33m"[39;49;00m])
        parser.add_argument([33m"[39;49;00m[33m--model-dir[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_MODEL_DIR[39;49;00m[33m"[39;49;00m])
        parser.add_argument([33m"[39;49;00m[33m--data-dir[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_CHANNEL_TRAINING[39;49;00m[33m"[39;49;00m])
        parser.add_argument([33m"[39;49;00m[33m--test[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_CHANNEL_TESTING[39;49;00m[33m"[39;49;00m])
        parser.add_argument([33m"[39;49;00m[33m--num-gpus[39;49;00m[33m"[39;49;00m, [36mtype[39;49;00m=[36mint[39;49;00m, default=os.environ[[33m"[39;49;00m[33mSM_NUM_GPUS[39;49;00m[33m"[39;49;00m])
    
        train(parser.parse_args())



```python
# Train on Amazon SageMaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train_deploy.py",
    source_dir="code",
    role=role,
    framework_version="1.5.0",
    py_version="py3",
    instance_count=1,  # this script only support distributed training for GPU instances.
    instance_type="local",
    hyperparameters={
        "epochs": 1,
        "num_labels": 2,
        "backend": "gloo",
    }
)
estimator.fit({"training": inputs_train, "testing": inputs_test})
```

    Creating tmpvrcwx0eo_algo-1-qz8z7_1 ... 
    [1BAttaching to tmpvrcwx0eo_algo-1-qz8z7_12mdone[0m
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:05:57,788 sagemaker-containers INFO     Imported framework sagemaker_pytorch_container.training
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:05:57,791 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:05:57,801 sagemaker_pytorch_container.training INFO     Block until all host DNS lookups succeed.
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:05:57,804 sagemaker_pytorch_container.training INFO     Invoking user training script.
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:05:58,926 sagemaker-containers INFO     Module default_user_module_name does not provide a setup.py. 
    [36malgo-1-qz8z7_1  |[0m Generating setup.py
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:05:58,927 sagemaker-containers INFO     Generating setup.cfg
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:05:58,927 sagemaker-containers INFO     Generating MANIFEST.in
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:05:58,927 sagemaker-containers INFO     Installing module with the following command:
    [36malgo-1-qz8z7_1  |[0m /opt/conda/bin/python -m pip install . -r requirements.txt
    [36malgo-1-qz8z7_1  |[0m Processing /tmp/tmpxfqtj2h0/module_dir
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: tqdm in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 1)) (4.42.1)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: requests==2.22.0 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 2)) (2.22.0)
    [36malgo-1-qz8z7_1  |[0m Collecting regex
    [36malgo-1-qz8z7_1  |[0m   Downloading regex-2020.11.13-cp36-cp36m-manylinux2014_x86_64.whl (723 kB)
    [K     |████████████████████████████████| 723 kB 19.0 MB/s eta 0:00:01
    [36malgo-1-qz8z7_1  |[0m [?25hCollecting sentencepiece
    [36malgo-1-qz8z7_1  |[0m   Downloading sentencepiece-0.1.94-cp36-cp36m-manylinux2014_x86_64.whl (1.1 MB)
    [K     |████████████████████████████████| 1.1 MB 84.2 MB/s eta 0:00:01
    [36malgo-1-qz8z7_1  |[0m [?25hCollecting sacremoses
    [36malgo-1-qz8z7_1  |[0m   Downloading sacremoses-0.0.43.tar.gz (883 kB)
    [K     |████████████████████████████████| 883 kB 76.3 MB/s eta 0:00:01
    [36malgo-1-qz8z7_1  |[0m [?25hCollecting transformers==2.3.0
    [36malgo-1-qz8z7_1  |[0m   Downloading transformers-2.3.0-py3-none-any.whl (447 kB)
    [K     |████████████████████████████████| 447 kB 81.8 MB/s eta 0:00:01
    [36malgo-1-qz8z7_1  |[0m [?25hRequirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r requirements.txt (line 2)) (1.25.8)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r requirements.txt (line 2)) (2.8)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r requirements.txt (line 2)) (2020.4.5.1)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r requirements.txt (line 2)) (3.0.4)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: six in /opt/conda/lib/python3.6/site-packages (from sacremoses->-r requirements.txt (line 5)) (1.14.0)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: click in /opt/conda/lib/python3.6/site-packages (from sacremoses->-r requirements.txt (line 5)) (7.1.2)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: joblib in /opt/conda/lib/python3.6/site-packages (from sacremoses->-r requirements.txt (line 5)) (0.14.1)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: boto3 in /opt/conda/lib/python3.6/site-packages (from transformers==2.3.0->-r requirements.txt (line 6)) (1.13.1)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: numpy in /opt/conda/lib/python3.6/site-packages (from transformers==2.3.0->-r requirements.txt (line 6)) (1.16.4)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: botocore<1.17.0,>=1.16.1 in /opt/conda/lib/python3.6/site-packages (from boto3->transformers==2.3.0->-r requirements.txt (line 6)) (1.16.1)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /opt/conda/lib/python3.6/site-packages (from boto3->transformers==2.3.0->-r requirements.txt (line 6)) (0.9.5)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /opt/conda/lib/python3.6/site-packages (from boto3->transformers==2.3.0->-r requirements.txt (line 6)) (0.3.3)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /opt/conda/lib/python3.6/site-packages (from botocore<1.17.0,>=1.16.1->boto3->transformers==2.3.0->-r requirements.txt (line 6)) (2.8.1)
    [36malgo-1-qz8z7_1  |[0m Requirement already satisfied: docutils<0.16,>=0.10 in /opt/conda/lib/python3.6/site-packages (from botocore<1.17.0,>=1.16.1->boto3->transformers==2.3.0->-r requirements.txt (line 6)) (0.15.2)
    [36malgo-1-qz8z7_1  |[0m Building wheels for collected packages: sacremoses, default-user-module-name
    [36malgo-1-qz8z7_1  |[0m   Building wheel for sacremoses (setup.py) ... [?25ldone
    [36malgo-1-qz8z7_1  |[0m [?25h  Created wheel for sacremoses: filename=sacremoses-0.0.43-py3-none-any.whl size=893259 sha256=55652629f4212554c1b0793a7ac31b0d34a083917063216596e45c54944e77ad
    [36malgo-1-qz8z7_1  |[0m   Stored in directory: /root/.cache/pip/wheels/49/25/98/cdea9c79b2d9a22ccc59540b1784b67f06b633378e97f58da2
    [36malgo-1-qz8z7_1  |[0m   Building wheel for default-user-module-name (setup.py) ... [?25ldone
    [36malgo-1-qz8z7_1  |[0m [?25h  Created wheel for default-user-module-name: filename=default_user_module_name-1.0.0-py2.py3-none-any.whl size=16534 sha256=b7d85b58838b91280c0d1edf5fad7cf63dcddeba95aa7d48d084a1267c1a6c02
    [36malgo-1-qz8z7_1  |[0m   Stored in directory: /tmp/pip-ephem-wheel-cache-nw1x3njf/wheels/5b/50/d2/e42b3b7af6205f28fcd930f210f37adaac67f544d70d9a693c
    [36malgo-1-qz8z7_1  |[0m Successfully built sacremoses default-user-module-name
    [36malgo-1-qz8z7_1  |[0m Installing collected packages: regex, sentencepiece, sacremoses, transformers, default-user-module-name
    [36malgo-1-qz8z7_1  |[0m Successfully installed default-user-module-name-1.0.0 regex-2020.11.13 sacremoses-0.0.43 sentencepiece-0.1.94 transformers-2.3.0
    [36malgo-1-qz8z7_1  |[0m [33mWARNING: You are using pip version 20.1; however, version 20.3.3 is available.
    [36malgo-1-qz8z7_1  |[0m You should consider upgrading via the '/opt/conda/bin/python -m pip install --upgrade pip' command.[0m
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:06:02,676 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:06:02,689 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:06:02,702 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:06:02,713 sagemaker-containers INFO     Invoking user script
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m Training Env:
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m {
    [36malgo-1-qz8z7_1  |[0m     "additional_framework_parameters": {},
    [36malgo-1-qz8z7_1  |[0m     "channel_input_dirs": {
    [36malgo-1-qz8z7_1  |[0m         "training": "/opt/ml/input/data/training",
    [36malgo-1-qz8z7_1  |[0m         "testing": "/opt/ml/input/data/testing"
    [36malgo-1-qz8z7_1  |[0m     },
    [36malgo-1-qz8z7_1  |[0m     "current_host": "algo-1-qz8z7",
    [36malgo-1-qz8z7_1  |[0m     "framework_module": "sagemaker_pytorch_container.training:main",
    [36malgo-1-qz8z7_1  |[0m     "hosts": [
    [36malgo-1-qz8z7_1  |[0m         "algo-1-qz8z7"
    [36malgo-1-qz8z7_1  |[0m     ],
    [36malgo-1-qz8z7_1  |[0m     "hyperparameters": {
    [36malgo-1-qz8z7_1  |[0m         "epochs": 1,
    [36malgo-1-qz8z7_1  |[0m         "num_labels": 2,
    [36malgo-1-qz8z7_1  |[0m         "backend": "gloo"
    [36malgo-1-qz8z7_1  |[0m     },
    [36malgo-1-qz8z7_1  |[0m     "input_config_dir": "/opt/ml/input/config",
    [36malgo-1-qz8z7_1  |[0m     "input_data_config": {
    [36malgo-1-qz8z7_1  |[0m         "training": {
    [36malgo-1-qz8z7_1  |[0m             "TrainingInputMode": "File"
    [36malgo-1-qz8z7_1  |[0m         },
    [36malgo-1-qz8z7_1  |[0m         "testing": {
    [36malgo-1-qz8z7_1  |[0m             "TrainingInputMode": "File"
    [36malgo-1-qz8z7_1  |[0m         }
    [36malgo-1-qz8z7_1  |[0m     },
    [36malgo-1-qz8z7_1  |[0m     "input_dir": "/opt/ml/input",
    [36malgo-1-qz8z7_1  |[0m     "is_master": true,
    [36malgo-1-qz8z7_1  |[0m     "job_name": "pytorch-training-2020-12-17-09-04-40-915",
    [36malgo-1-qz8z7_1  |[0m     "log_level": 20,
    [36malgo-1-qz8z7_1  |[0m     "master_hostname": "algo-1-qz8z7",
    [36malgo-1-qz8z7_1  |[0m     "model_dir": "/opt/ml/model",
    [36malgo-1-qz8z7_1  |[0m     "module_dir": "s3://sagemaker-us-east-1-302844384390/pytorch-training-2020-12-17-09-04-40-915/source/sourcedir.tar.gz",
    [36malgo-1-qz8z7_1  |[0m     "module_name": "train_deploy",
    [36malgo-1-qz8z7_1  |[0m     "network_interface_name": "eth0",
    [36malgo-1-qz8z7_1  |[0m     "num_cpus": 8,
    [36malgo-1-qz8z7_1  |[0m     "num_gpus": 0,
    [36malgo-1-qz8z7_1  |[0m     "output_data_dir": "/opt/ml/output/data",
    [36malgo-1-qz8z7_1  |[0m     "output_dir": "/opt/ml/output",
    [36malgo-1-qz8z7_1  |[0m     "output_intermediate_dir": "/opt/ml/output/intermediate",
    [36malgo-1-qz8z7_1  |[0m     "resource_config": {
    [36malgo-1-qz8z7_1  |[0m         "current_host": "algo-1-qz8z7",
    [36malgo-1-qz8z7_1  |[0m         "hosts": [
    [36malgo-1-qz8z7_1  |[0m             "algo-1-qz8z7"
    [36malgo-1-qz8z7_1  |[0m         ]
    [36malgo-1-qz8z7_1  |[0m     },
    [36malgo-1-qz8z7_1  |[0m     "user_entry_point": "train_deploy.py"
    [36malgo-1-qz8z7_1  |[0m }
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m Environment variables:
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m SM_HOSTS=["algo-1-qz8z7"]
    [36malgo-1-qz8z7_1  |[0m SM_NETWORK_INTERFACE_NAME=eth0
    [36malgo-1-qz8z7_1  |[0m SM_HPS={"backend":"gloo","epochs":1,"num_labels":2}
    [36malgo-1-qz8z7_1  |[0m SM_USER_ENTRY_POINT=train_deploy.py
    [36malgo-1-qz8z7_1  |[0m SM_FRAMEWORK_PARAMS={}
    [36malgo-1-qz8z7_1  |[0m SM_RESOURCE_CONFIG={"current_host":"algo-1-qz8z7","hosts":["algo-1-qz8z7"]}
    [36malgo-1-qz8z7_1  |[0m SM_INPUT_DATA_CONFIG={"testing":{"TrainingInputMode":"File"},"training":{"TrainingInputMode":"File"}}
    [36malgo-1-qz8z7_1  |[0m SM_OUTPUT_DATA_DIR=/opt/ml/output/data
    [36malgo-1-qz8z7_1  |[0m SM_CHANNELS=["testing","training"]
    [36malgo-1-qz8z7_1  |[0m SM_CURRENT_HOST=algo-1-qz8z7
    [36malgo-1-qz8z7_1  |[0m SM_MODULE_NAME=train_deploy
    [36malgo-1-qz8z7_1  |[0m SM_LOG_LEVEL=20
    [36malgo-1-qz8z7_1  |[0m SM_FRAMEWORK_MODULE=sagemaker_pytorch_container.training:main
    [36malgo-1-qz8z7_1  |[0m SM_INPUT_DIR=/opt/ml/input
    [36malgo-1-qz8z7_1  |[0m SM_INPUT_CONFIG_DIR=/opt/ml/input/config
    [36malgo-1-qz8z7_1  |[0m SM_OUTPUT_DIR=/opt/ml/output
    [36malgo-1-qz8z7_1  |[0m SM_NUM_CPUS=8
    [36malgo-1-qz8z7_1  |[0m SM_NUM_GPUS=0
    [36malgo-1-qz8z7_1  |[0m SM_MODEL_DIR=/opt/ml/model
    [36malgo-1-qz8z7_1  |[0m SM_MODULE_DIR=s3://sagemaker-us-east-1-302844384390/pytorch-training-2020-12-17-09-04-40-915/source/sourcedir.tar.gz
    [36malgo-1-qz8z7_1  |[0m SM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"testing":"/opt/ml/input/data/testing","training":"/opt/ml/input/data/training"},"current_host":"algo-1-qz8z7","framework_module":"sagemaker_pytorch_container.training:main","hosts":["algo-1-qz8z7"],"hyperparameters":{"backend":"gloo","epochs":1,"num_labels":2},"input_config_dir":"/opt/ml/input/config","input_data_config":{"testing":{"TrainingInputMode":"File"},"training":{"TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"pytorch-training-2020-12-17-09-04-40-915","log_level":20,"master_hostname":"algo-1-qz8z7","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-us-east-1-302844384390/pytorch-training-2020-12-17-09-04-40-915/source/sourcedir.tar.gz","module_name":"train_deploy","network_interface_name":"eth0","num_cpus":8,"num_gpus":0,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1-qz8z7","hosts":["algo-1-qz8z7"]},"user_entry_point":"train_deploy.py"}
    [36malgo-1-qz8z7_1  |[0m SM_USER_ARGS=["--backend","gloo","--epochs","1","--num_labels","2"]
    [36malgo-1-qz8z7_1  |[0m SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
    [36malgo-1-qz8z7_1  |[0m SM_CHANNEL_TRAINING=/opt/ml/input/data/training
    [36malgo-1-qz8z7_1  |[0m SM_CHANNEL_TESTING=/opt/ml/input/data/testing
    [36malgo-1-qz8z7_1  |[0m SM_HP_EPOCHS=1
    [36malgo-1-qz8z7_1  |[0m SM_HP_NUM_LABELS=2
    [36malgo-1-qz8z7_1  |[0m SM_HP_BACKEND=gloo
    [36malgo-1-qz8z7_1  |[0m PYTHONPATH=/opt/ml/code:/opt/conda/bin:/opt/conda/lib/python36.zip:/opt/conda/lib/python3.6:/opt/conda/lib/python3.6/lib-dynload:/opt/conda/lib/python3.6/site-packages
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m Invoking script with the following command:
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m /opt/conda/bin/python train_deploy.py --backend gloo --epochs 1 --num_labels 2
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m Loading BERT tokenizer...
    [36malgo-1-qz8z7_1  |[0m Distributed training - False
    [36malgo-1-qz8z7_1  |[0m Number of gpus available - 0
    [36malgo-1-qz8z7_1  |[0m Get train data loader
    [36malgo-1-qz8z7_1  |[0m Processes 6413/6413 (100%) of train data
    [36malgo-1-qz8z7_1  |[0m Processes 2138/2138 (100%) of test data
    [36malgo-1-qz8z7_1  |[0m Starting BertForSequenceClassification
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m End of defining BertForSequenceClassification
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m Train Epoch: 1 [0/6413 (0%)] Loss: 0.647636
    [36malgo-1-qz8z7_1  |[0m Train Epoch: 1 [3200/6413 (50%)] Loss: 0.518038
    [36malgo-1-qz8z7_1  |[0m Train Epoch: 1 [1300/6413 (99%)] Loss: 0.598134
    [36malgo-1-qz8z7_1  |[0m Average training loss: 0.511798
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m Test set: Accuracy: 0.782609
    [36malgo-1-qz8z7_1  |[0m 
    [36malgo-1-qz8z7_1  |[0m Saving tuned model.
    [36malgo-1-qz8z7_1  |[0m 2020-12-17 09:19:10,283 sagemaker-containers INFO     Reporting training SUCCESS
    [36mtmpvrcwx0eo_algo-1-qz8z7_1 exited with code 0
    [0mAborting on container exit...
    ===== Job Complete =====



```python
# Host

predictor = estimator.deploy(initial_instance_count=1, instance_type='local')#, accelerator_type='local_sagemaker_notebook')

```

    Attaching to tmpspwlmnz0_algo-1-903iv_1
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: tqdm in /opt/conda/lib/python3.6/site-packages (from -r /opt/ml/model/code/requirements.txt (line 1)) (4.45.0)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: requests==2.22.0 in /opt/conda/lib/python3.6/site-packages (from -r /opt/ml/model/code/requirements.txt (line 2)) (2.22.0)
    [36malgo-1-903iv_1  |[0m Collecting regex
    [36malgo-1-903iv_1  |[0m   Downloading regex-2020.11.13-cp36-cp36m-manylinux2014_x86_64.whl (723 kB)
    [K     |████████████████████████████████| 723 kB 17.5 MB/s eta 0:00:01
    [36malgo-1-903iv_1  |[0m [?25hCollecting sentencepiece
    [36malgo-1-903iv_1  |[0m   Downloading sentencepiece-0.1.94-cp36-cp36m-manylinux2014_x86_64.whl (1.1 MB)
    [K     |████████████████████████████████| 1.1 MB 77.7 MB/s eta 0:00:01
    [36malgo-1-903iv_1  |[0m [?25hCollecting sacremoses
    [36malgo-1-903iv_1  |[0m   Downloading sacremoses-0.0.43.tar.gz (883 kB)
    [K     |████████████████████████████████| 883 kB 62.8 MB/s eta 0:00:01
    [36malgo-1-903iv_1  |[0m [?25hCollecting transformers==2.3.0
    [36malgo-1-903iv_1  |[0m   Downloading transformers-2.3.0-py3-none-any.whl (447 kB)
    [K     |████████████████████████████████| 447 kB 60.3 MB/s eta 0:00:01
    [36malgo-1-903iv_1  |[0m [?25hRequirement already satisfied: idna<2.9,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r /opt/ml/model/code/requirements.txt (line 2)) (2.8)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r /opt/ml/model/code/requirements.txt (line 2)) (3.0.4)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r /opt/ml/model/code/requirements.txt (line 2)) (1.25.8)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r /opt/ml/model/code/requirements.txt (line 2)) (2020.4.5.1)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: six in /opt/conda/lib/python3.6/site-packages (from sacremoses->-r /opt/ml/model/code/requirements.txt (line 5)) (1.14.0)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: click in /opt/conda/lib/python3.6/site-packages (from sacremoses->-r /opt/ml/model/code/requirements.txt (line 5)) (7.1.2)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: joblib in /opt/conda/lib/python3.6/site-packages (from sacremoses->-r /opt/ml/model/code/requirements.txt (line 5)) (0.14.1)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: numpy in /opt/conda/lib/python3.6/site-packages (from transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (1.16.4)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: boto3 in /opt/conda/lib/python3.6/site-packages (from transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (1.13.1)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: botocore<1.17.0,>=1.16.1 in /opt/conda/lib/python3.6/site-packages (from boto3->transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (1.16.1)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /opt/conda/lib/python3.6/site-packages (from boto3->transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (0.3.3)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /opt/conda/lib/python3.6/site-packages (from boto3->transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (0.9.5)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /opt/conda/lib/python3.6/site-packages (from botocore<1.17.0,>=1.16.1->boto3->transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (2.8.1)
    [36malgo-1-903iv_1  |[0m Requirement already satisfied: docutils<0.16,>=0.10 in /opt/conda/lib/python3.6/site-packages (from botocore<1.17.0,>=1.16.1->boto3->transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (0.15.2)
    [36malgo-1-903iv_1  |[0m Building wheels for collected packages: sacremoses
    [36malgo-1-903iv_1  |[0m   Building wheel for sacremoses (setup.py) ... [?25ldone
    [36malgo-1-903iv_1  |[0m [?25h  Created wheel for sacremoses: filename=sacremoses-0.0.43-py3-none-any.whl size=893259 sha256=387512783cac3f065514dbbc5861bf6b9327666b9123e1266d3fc865f4d7d256
    [36malgo-1-903iv_1  |[0m   Stored in directory: /root/.cache/pip/wheels/49/25/98/cdea9c79b2d9a22ccc59540b1784b67f06b633378e97f58da2
    [36malgo-1-903iv_1  |[0m Successfully built sacremoses
    [36malgo-1-903iv_1  |[0m Installing collected packages: regex, sentencepiece, sacremoses, transformers
    [36malgo-1-903iv_1  |[0m Successfully installed regex-2020.11.13 sacremoses-0.0.43 sentencepiece-0.1.94 transformers-2.3.0
    [36malgo-1-903iv_1  |[0m [33mWARNING: You are using pip version 20.1; however, version 20.3.3 is available.
    [36malgo-1-903iv_1  |[0m You should consider upgrading via the '/opt/conda/bin/python -m pip install --upgrade pip' command.[0m
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,732 [INFO ] main com.amazonaws.ml.mms.ModelServer - 
    [36malgo-1-903iv_1  |[0m MMS Home: /opt/conda/lib/python3.6/site-packages
    [36malgo-1-903iv_1  |[0m Current directory: /
    [36malgo-1-903iv_1  |[0m Temp directory: /home/model-server/tmp
    [36malgo-1-903iv_1  |[0m Number of GPUs: 0
    [36malgo-1-903iv_1  |[0m Number of CPUs: 8
    [36malgo-1-903iv_1  |[0m Max heap size: 6972 M
    [36malgo-1-903iv_1  |[0m Python executable: /opt/conda/bin/python
    [36malgo-1-903iv_1  |[0m Config file: /etc/sagemaker-mms.properties
    [36malgo-1-903iv_1  |[0m Inference address: http://0.0.0.0:8080
    [36malgo-1-903iv_1  |[0m Management address: http://0.0.0.0:8080
    [36malgo-1-903iv_1  |[0m Model Store: /.sagemaker/mms/models
    [36malgo-1-903iv_1  |[0m Initial Models: ALL
    [36malgo-1-903iv_1  |[0m Log dir: /logs
    [36malgo-1-903iv_1  |[0m Metrics dir: /logs
    [36malgo-1-903iv_1  |[0m Netty threads: 0
    [36malgo-1-903iv_1  |[0m Netty client threads: 0
    [36malgo-1-903iv_1  |[0m Default workers per model: 8
    [36malgo-1-903iv_1  |[0m Blacklist Regex: N/A
    [36malgo-1-903iv_1  |[0m Maximum Response Size: 6553500
    [36malgo-1-903iv_1  |[0m Maximum Request Size: 6553500
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,774 [INFO ] main com.amazonaws.ml.mms.wlm.ModelManager - Model model loaded.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,789 [INFO ] main com.amazonaws.ml.mms.ModelServer - Initialize Inference server with: EpollServerSocketChannel.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,934 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Listening on port: /home/model-server/tmp/.mms.sock.9007
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,935 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [PID]62
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,942 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Listening on port: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,942 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [PID]59
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,943 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - MXNet worker started.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,943 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Python runtime: 3.6.6
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,943 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - MXNet worker started.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,943 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Python runtime: 3.6.6
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,943 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Listening on port: /home/model-server/tmp/.mms.sock.9005
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,944 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [PID]63
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,944 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Listening on port: /home/model-server/tmp/.mms.sock.9004
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,945 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [PID]60
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,945 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - MXNet worker started.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,945 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Python runtime: 3.6.6
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,945 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - MXNet worker started.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,946 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Python runtime: 3.6.6
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,948 [INFO ] W-9005-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9005
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,948 [INFO ] W-9007-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9007
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,949 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,949 [INFO ] W-9004-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9004
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,953 [INFO ] main com.amazonaws.ml.mms.ModelServer - Inference API bind to: http://0.0.0.0:8080
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,965 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9005.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,965 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9004.
    [36malgo-1-903iv_1  |[0m Model server started.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,966 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Listening on port: /home/model-server/tmp/.mms.sock.9001
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,967 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [PID]61
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,967 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - MXNet worker started.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,965 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9007.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,967 [INFO ] W-9001-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9001
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,965 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,967 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Python runtime: 3.6.6
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,972 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9001.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,976 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Listening on port: /home/model-server/tmp/.mms.sock.9006
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,976 [WARN ] pool-2-thread-1 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,976 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [PID]58
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,976 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - MXNet worker started.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,976 [INFO ] W-9006-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9006
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,976 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Python runtime: 3.6.6
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,980 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9006.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,980 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Listening on port: /home/model-server/tmp/.mms.sock.9003
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,981 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [PID]65
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,981 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - MXNet worker started.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,981 [INFO ] W-9003-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9003
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,981 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Python runtime: 3.6.6
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,984 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Listening on port: /home/model-server/tmp/.mms.sock.9002
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,984 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9003.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,984 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [PID]57
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,985 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - MXNet worker started.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,985 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Python runtime: 3.6.6
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,985 [INFO ] W-9002-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9002
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:08,988 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9002.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,251 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.5.0+cpu available.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,251 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.5.0+cpu available.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,270 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.5.0+cpu available.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,270 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.5.0+cpu available.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,311 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.5.0+cpu available.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,311 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.5.0+cpu available.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,316 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.5.0+cpu available.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,329 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.5.0+cpu available.
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,715 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,715 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmpm8uycn48
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,729 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,730 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmp_cf3o656
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,736 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,736 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmprv54ia36
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,751 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,751 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmps7mmrzl_
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,763 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,763 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmp5vss138p
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,765 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmpm8uycn48 to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,766 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmp_cf3o656 to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,766 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,766 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,767 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,767 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmpm8uycn48
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,767 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,768 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,768 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmp_cf3o656
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,770 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,786 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmprv54ia36 to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,787 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,787 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,787 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,788 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmprv54ia36
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,788 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,791 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,791 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,792 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmps7mmrzl_ to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,794 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,794 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmps7mmrzl_
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,794 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,796 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading configuration file /.sagemaker/mms/models/model/config.json
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,797 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Model config {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,798 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "architectures": [
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,798 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "BertForMaskedLM"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,798 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   ],
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,799 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "attention_probs_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,799 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "finetuning_task": null,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,800 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_act": "gelu",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,800 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,800 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_size": 768,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,801 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "id2label": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,801 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "0": "LABEL_0",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,802 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "1": "LABEL_1"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,802 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,802 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "initializer_range": 0.02,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,803 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "intermediate_size": 3072,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,803 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "is_decoder": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,804 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "label2id": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,804 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_0": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,804 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_1": 1
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,805 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,805 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "layer_norm_eps": 1e-12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,806 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "max_position_embeddings": 512,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,806 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "model_type": "bert",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,806 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading configuration file /.sagemaker/mms/models/model/config.json
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,806 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_attention_heads": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,807 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_hidden_layers": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,807 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Model config {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,807 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "architectures": [
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,807 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_labels": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,808 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "BertForMaskedLM"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,809 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading configuration file /.sagemaker/mms/models/model/config.json
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,811 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   ],
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,808 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_attentions": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,812 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Model config {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,812 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "architectures": [
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,812 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "attention_probs_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,813 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "finetuning_task": null,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,813 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_act": "gelu",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,813 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading configuration file /.sagemaker/mms/models/model/config.json
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,814 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Model config {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,814 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "architectures": [
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,814 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "BertForMaskedLM"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,814 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   ],
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,814 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "attention_probs_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,814 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "finetuning_task": null,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,814 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_act": "gelu",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,814 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_size": 768,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "id2label": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "0": "LABEL_0",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "1": "LABEL_1"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "initializer_range": 0.02,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "intermediate_size": 3072,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "is_decoder": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "label2id": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,815 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_0": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_1": 1
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "layer_norm_eps": 1e-12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "max_position_embeddings": 512,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "model_type": "bert",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_attention_heads": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_hidden_layers": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_labels": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_attentions": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_hidden_states": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_past": true,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,816 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pad_token_id": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,817 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pruned_heads": {},
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,817 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "torchscript": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,817 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "type_vocab_size": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,817 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "use_bfloat16": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,817 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "vocab_size": 30522
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,817 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - }
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,817 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - 
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,817 [INFO ] W-9006-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading weights file /.sagemaker/mms/models/model/pytorch_model.bin
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,812 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_hidden_states": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,819 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_past": true,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,813 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,813 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "BertForMaskedLM"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,819 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_size": 768,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,820 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "id2label": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,820 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "0": "LABEL_0",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,820 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "1": "LABEL_1"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,821 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,821 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "initializer_range": 0.02,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,821 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "intermediate_size": 3072,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,821 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "is_decoder": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,821 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "label2id": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,822 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_0": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,820 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   ],
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,822 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "attention_probs_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,822 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "finetuning_task": null,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,822 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_1": 1
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,823 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,823 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "layer_norm_eps": 1e-12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,823 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_act": "gelu",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,824 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,824 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_size": 768,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,824 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "max_position_embeddings": 512,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,824 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "id2label": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,825 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "model_type": "bert",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,825 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_attention_heads": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,825 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "0": "LABEL_0",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,825 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "1": "LABEL_1"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,825 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_hidden_layers": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,826 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,826 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "initializer_range": 0.02,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,826 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_labels": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,827 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "intermediate_size": 3072,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,827 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "is_decoder": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,827 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_attentions": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,827 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "label2id": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,827 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_hidden_states": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,828 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_0": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,828 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_1": 1
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,828 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_past": true,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,828 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pad_token_id": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,828 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,829 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "layer_norm_eps": 1e-12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,829 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pruned_heads": {},
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,829 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "torchscript": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,830 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "type_vocab_size": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,831 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "use_bfloat16": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,829 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "max_position_embeddings": 512,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,830 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading configuration file /.sagemaker/mms/models/model/config.json
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,831 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "vocab_size": 30522
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,839 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "model_type": "bert",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,838 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmp5vss138p to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,840 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_attention_heads": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,835 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading configuration file /.sagemaker/mms/models/model/config.json
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,840 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_hidden_layers": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,841 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading configuration file /.sagemaker/mms/models/model/config.json
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,843 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_labels": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,840 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Model config {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,843 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_attentions": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,840 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Model config {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,843 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_hidden_states": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "architectures": [
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "BertForMaskedLM"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,840 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - }
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,840 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   ],
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Model config {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "architectures": [
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "BertForMaskedLM"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   ],
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "attention_probs_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "finetuning_task": null,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_act": "gelu",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_size": 768,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "architectures": [
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - 
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "id2label": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmp5vss138p
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "0": "LABEL_0",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "1": "LABEL_1"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "attention_probs_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "finetuning_task": null,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_act": "gelu",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_size": 768,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "id2label": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "0": "LABEL_0",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "1": "LABEL_1"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "initializer_range": 0.02,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "intermediate_size": 3072,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "is_decoder": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "label2id": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_0": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_1": 1
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "layer_norm_eps": 1e-12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "max_position_embeddings": 512,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "model_type": "bert",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_attention_heads": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_hidden_layers": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_labels": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_attentions": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_hidden_states": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,846 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_past": true,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pad_token_id": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pruned_heads": {},
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "torchscript": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "type_vocab_size": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "use_bfloat16": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "vocab_size": 30522
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - }
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - 
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9003-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading weights file /.sagemaker/mms/models/model/pytorch_model.bin
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,844 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_past": true,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pad_token_id": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pruned_heads": {},
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "torchscript": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "type_vocab_size": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "use_bfloat16": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "vocab_size": 30522
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,847 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - }
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - 
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9002-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading weights file /.sagemaker/mms/models/model/pytorch_model.bin
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "initializer_range": 0.02,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "intermediate_size": 3072,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "is_decoder": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "label2id": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_0": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_1": 1
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "layer_norm_eps": 1e-12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "max_position_embeddings": 512,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "model_type": "bert",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_attention_heads": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_hidden_layers": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_labels": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,848 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_attentions": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_hidden_states": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_past": true,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pad_token_id": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pruned_heads": {},
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "torchscript": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "type_vocab_size": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "use_bfloat16": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "vocab_size": 30522
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - }
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - 
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9001-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading weights file /.sagemaker/mms/models/model/pytorch_model.bin
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9007-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading weights file /.sagemaker/mms/models/model/pytorch_model.bin
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,845 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "BertForMaskedLM"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   ],
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,849 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "attention_probs_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "finetuning_task": null,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_act": "gelu",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_size": 768,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "id2label": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "0": "LABEL_0",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "1": "LABEL_1"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "initializer_range": 0.02,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "intermediate_size": 3072,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "is_decoder": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "label2id": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_0": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_1": 1
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "layer_norm_eps": 1e-12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "max_position_embeddings": 512,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,850 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "model_type": "bert",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_attention_heads": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_hidden_layers": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_labels": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_attentions": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_hidden_states": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_past": true,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pad_token_id": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pruned_heads": {},
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "torchscript": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "type_vocab_size": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "use_bfloat16": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "vocab_size": 30522
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - }
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - 
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,851 [INFO ] W-9004-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading weights file /.sagemaker/mms/models/model/pytorch_model.bin
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,819 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pad_token_id": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,857 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pruned_heads": {},
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,858 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "torchscript": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,858 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "type_vocab_size": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,858 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "use_bfloat16": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,858 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "vocab_size": 30522
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,858 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - }
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,858 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - 
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,859 [INFO ] W-9005-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading weights file /.sagemaker/mms/models/model/pytorch_model.bin
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,860 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,906 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading configuration file /.sagemaker/mms/models/model/config.json
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,907 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Model config {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,907 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "architectures": [
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,908 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "BertForMaskedLM"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,908 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   ],
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,908 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "attention_probs_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,909 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "finetuning_task": null,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,909 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_act": "gelu",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,909 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_dropout_prob": 0.1,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,909 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "hidden_size": 768,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,909 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "id2label": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,910 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "0": "LABEL_0",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,910 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "1": "LABEL_1"
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,910 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,910 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "initializer_range": 0.02,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,918 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "intermediate_size": 3072,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,918 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "is_decoder": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,918 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "label2id": {
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,918 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_0": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,918 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -     "LABEL_1": 1
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,918 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   },
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "layer_norm_eps": 1e-12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "max_position_embeddings": 512,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "model_type": "bert",
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_attention_heads": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_hidden_layers": 12,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "num_labels": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_attentions": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_hidden_states": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "output_past": true,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pad_token_id": 0,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "pruned_heads": {},
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "torchscript": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "type_vocab_size": 2,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "use_bfloat16": false,
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle -   "vocab_size": 30522
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - }
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - 
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:10,919 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading weights file /.sagemaker/mms/models/model/pytorch_model.bin
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:11,959 [INFO ] pool-1-thread-9 ACCESS_LOG - /172.18.0.1:45866 "GET /ping HTTP/1.1" 200 38
    ![36malgo-1-903iv_1  |[0m 2020-12-17 09:23:16,361 [INFO ] W-9004-model com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 7362
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:16,366 [INFO ] W-9005-model com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 7363
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:16,389 [INFO ] W-9002-model com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 7385
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:16,389 [INFO ] W-9001-model com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 7385
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:16,391 [INFO ] W-9007-model com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 7387
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:16,391 [INFO ] W-9003-model com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 7388
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:16,401 [INFO ] W-9006-model com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 7397
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:16,417 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 7413



```python
#from sagemaker.predictor import json_deserializer, json_serializer

#predictor.content_type = "application/json"
#predictor.accept = "application/json"
predictor.serializer = sagemaker.serializers.JSONSerializer()
predictor.deserializer = sagemaker.deserializers.JSONDeserializer()
```


```python
result = predictor.predict("Somebody just left - guess who.")
print(np.argmax(result, axis=1))
```

    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:54,573 [INFO ] W-9004-model com.amazonaws.ml.mms.wlm.WorkerThread - Backend response time: 180
    [36malgo-1-903iv_1  |[0m 2020-12-17 09:23:54,573 [INFO ] W-9004-model ACCESS_LOG - /172.18.0.1:45918 "POST /invocations HTTP/1.1" 200 183
    [1]



```python
predictor.delete_endpoint()
```

    Gracefully stopping... (press Ctrl+C again to force)



```python
estimator.model_data
```




    's3://sagemaker-us-east-1-302844384390/pytorch-training-2020-12-17-09-04-40-915/model.tar.gz'




```python
!rm -r model
```


```sh
%%sh -s $estimator.model_data
mkdir model
aws s3 cp $1 model/ 
tar xvzf model/model.tar.gz --directory ./model
```

    download: s3://sagemaker-us-east-1-302844384390/pytorch-training-2020-12-17-09-04-40-915/model.tar.gz to model/model.tar.gz
    pytorch_model.bin
    config.json



```python
# The following code converts our model into the TorchScript format:
!pip install transformers==2.11.0
import subprocess
import torch
from transformers import BertForSequenceClassification

model_torchScript = BertForSequenceClassification.from_pretrained("model/", torchscript=True)
device = "cpu"
for_jit_trace_input_ids = [0] * 64
for_jit_trace_attention_masks = [0] * 64
for_jit_trace_input = torch.tensor([for_jit_trace_input_ids])
for_jit_trace_masks = torch.tensor([for_jit_trace_input_ids])

traced_model = torch.jit.trace(
    model_torchScript, [for_jit_trace_input.to(device), for_jit_trace_masks.to(device)]
)
torch.jit.save(traced_model, "traced_bert.pt")

subprocess.call(["tar", "-czvf", "traced_bert.tar.gz", "traced_bert.pt"])
```

    Collecting transformers==2.11.0
      Downloading transformers-2.11.0-py3-none-any.whl (674 kB)
    [K     |████████████████████████████████| 674 kB 19.7 MB/s eta 0:00:01
    [?25hCollecting tokenizers==0.7.0
      Downloading tokenizers-0.7.0-cp36-cp36m-manylinux1_x86_64.whl (3.8 MB)
    [K     |████████████████████████████████| 3.8 MB 31.4 MB/s eta 0:00:01
    [?25hCollecting sentencepiece
      Downloading sentencepiece-0.1.94-cp36-cp36m-manylinux2014_x86_64.whl (1.1 MB)
    [K     |████████████████████████████████| 1.1 MB 81.7 MB/s eta 0:00:01
    [?25hRequirement already satisfied: sacremoses in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from transformers==2.11.0) (0.0.43)
    Requirement already satisfied: tqdm>=4.27 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from transformers==2.11.0) (4.42.1)
    Requirement already satisfied: regex!=2019.12.17 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from transformers==2.11.0) (2020.11.13)
    Requirement already satisfied: requests in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from transformers==2.11.0) (2.22.0)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from transformers==2.11.0) (0.7)
    Requirement already satisfied: packaging in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from transformers==2.11.0) (20.1)
    Requirement already satisfied: filelock in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from transformers==2.11.0) (3.0.12)
    Requirement already satisfied: numpy in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from transformers==2.11.0) (1.18.1)
    Requirement already satisfied: six in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sacremoses->transformers==2.11.0) (1.14.0)
    Requirement already satisfied: joblib in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sacremoses->transformers==2.11.0) (0.14.1)
    Requirement already satisfied: click in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from sacremoses->transformers==2.11.0) (7.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from requests->transformers==2.11.0) (1.25.10)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from requests->transformers==2.11.0) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from requests->transformers==2.11.0) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from requests->transformers==2.11.0) (2020.6.20)
    Requirement already satisfied: pyparsing>=2.0.2 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from packaging->transformers==2.11.0) (2.4.6)
    Installing collected packages: tokenizers, sentencepiece, transformers
      Attempting uninstall: tokenizers
        Found existing installation: tokenizers 0.9.4
        Uninstalling tokenizers-0.9.4:
          Successfully uninstalled tokenizers-0.9.4
      Attempting uninstall: transformers
        Found existing installation: transformers 4.0.1
        Uninstalling transformers-4.0.1:
          Successfully uninstalled transformers-4.0.1
    Successfully installed sentencepiece-0.1.94 tokenizers-0.7.0 transformers-2.11.0
    [33mWARNING: You are using pip version 20.0.2; however, version 20.3.3 is available.
    You should consider upgrading via the '/home/ec2-user/anaconda3/envs/pytorch_p36/bin/python -m pip install --upgrade pip' command.[0m
    [36malgo-1-q1fo1_1  |[0m 2020-12-17 09:51:10,414 [WARN ] pool-2-thread-2 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.





    0




```python
!pygmentize code/deploy_ei.py
```

    [34mimport[39;49;00m [04m[36mjson[39;49;00m
    [34mimport[39;49;00m [04m[36mlogging[39;49;00m
    [34mimport[39;49;00m [04m[36mos[39;49;00m
    [34mimport[39;49;00m [04m[36msys[39;49;00m
    
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36mutils[39;49;00m[04m[36m.[39;49;00m[04m[36mdata[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36mutils[39;49;00m[04m[36m.[39;49;00m[04m[36mdata[39;49;00m[04m[36m.[39;49;00m[04m[36mdistributed[39;49;00m
    [34mfrom[39;49;00m [04m[36mtransformers[39;49;00m [34mimport[39;49;00m BertTokenizer
    
    logger = logging.getLogger([31m__name__[39;49;00m)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    MAX_LEN = [34m64[39;49;00m  [37m# this is the max length of the sentence[39;49;00m
    
    [36mprint[39;49;00m([33m"[39;49;00m[33mLoading BERT tokenizer...[39;49;00m[33m"[39;49;00m)
    tokenizer = BertTokenizer.from_pretrained([33m"[39;49;00m[33mbert-base-uncased[39;49;00m[33m"[39;49;00m, do_lower_case=[34mTrue[39;49;00m)
    
    
    [34mdef[39;49;00m [32mmodel_fn[39;49;00m(model_dir):
        device = torch.device([33m"[39;49;00m[33mcuda[39;49;00m[33m"[39;49;00m [34mif[39;49;00m torch.cuda.is_available() [34melse[39;49;00m [33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m)
    
        loaded_model = torch.jit.load(os.path.join(model_dir, [33m"[39;49;00m[33mtraced_bert.pt[39;49;00m[33m"[39;49;00m))
        [34mreturn[39;49;00m loaded_model.to(device)
    
    
    [34mdef[39;49;00m [32minput_fn[39;49;00m(request_body, request_content_type):
        [33m"""An input_fn that loads a pickled tensor"""[39;49;00m
        [34mif[39;49;00m request_content_type == [33m"[39;49;00m[33mapplication/json[39;49;00m[33m"[39;49;00m:
            sentence = json.loads(request_body)
    
            input_ids = []
            encoded_sent = tokenizer.encode(sentence, add_special_tokens=[34mTrue[39;49;00m)
            input_ids.append(encoded_sent)
    
            [37m# pad shorter sentences[39;49;00m
            input_ids_padded = []
            [34mfor[39;49;00m i [35min[39;49;00m input_ids:
                [34mwhile[39;49;00m [36mlen[39;49;00m(i) < MAX_LEN:
                    i.append([34m0[39;49;00m)
                input_ids_padded.append(i)
            input_ids = input_ids_padded
    
            [37m# mask; 0: added, 1: otherwise[39;49;00m
            attention_masks = []
            [37m# For each sentence...[39;49;00m
            [34mfor[39;49;00m sent [35min[39;49;00m input_ids:
                att_mask = [[36mint[39;49;00m(token_id > [34m0[39;49;00m) [34mfor[39;49;00m token_id [35min[39;49;00m sent]
                attention_masks.append(att_mask)
    
            [37m# convert to PyTorch data types.[39;49;00m
            train_inputs = torch.tensor(input_ids)
            train_masks = torch.tensor(attention_masks)
    
            [34mreturn[39;49;00m train_inputs, train_masks
    
        [34mraise[39;49;00m [36mValueError[39;49;00m([33m"[39;49;00m[33mUnsupported content type: [39;49;00m[33m{}[39;49;00m[33m"[39;49;00m.format(request_content_type))
    
    
    [34mdef[39;49;00m [32mpredict_fn[39;49;00m(input_data, model):
        device = torch.device([33m"[39;49;00m[33mcuda[39;49;00m[33m"[39;49;00m [34mif[39;49;00m torch.cuda.is_available() [34melse[39;49;00m [33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m)
        model.to(device)
        model.eval()
    
        input_id, input_mask = input_data
        input_id = input_id.to(device)
        input_mask = input_mask.to(device)
        [34mwith[39;49;00m torch.no_grad():
            [34mwith[39;49;00m torch.jit.optimized_execution([34mTrue[39;49;00m, {[33m"[39;49;00m[33mtarget_device[39;49;00m[33m"[39;49;00m: [33m"[39;49;00m[33meia:0[39;49;00m[33m"[39;49;00m}):
                [34mreturn[39;49;00m model(input_id, attention_mask=input_mask)[[34m0[39;49;00m]



```python
role
```




    'arn:aws:iam::302844384390:role/service-role/hercule_training_role'




```python
sagemaker_session
```




    <sagemaker.local.local_session.LocalSession at 0x7f4e2e77c128>




```python
# Next we upload TorchScript model to S3 and deploy using Elastic Inference. 
# The accelerator_type=ml.eia2.xlarge parameter is how we attach the Elastic Inference accelerator to our endpoint.

from sagemaker.pytorch import PyTorchModel

instance_type = "local"#'ml.m5.large'
accelerator_type = "local_sagemaker_notebook"#'ml.eia2.xlarge'

# TorchScript model
tar_filename = 'traced_bert.tar.gz'

# Returns S3 bucket URL
print('Upload tarball to S3')
model_data = sagemaker_session.upload_data(path=tar_filename, bucket=bucket, key_prefix=prefix)

endpoint_name = 'bert-ei-traced-{}-{}'.format(instance_type, accelerator_type).replace('.', '').replace('_', '')

pytorch = PyTorchModel(
    model_data=model_data,
    role=role,
    entry_point='deploy_ei.py',
    source_dir='code',
    framework_version='1.3.1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

```

    Upload tarball to S3



```python
predictor = pytorch.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    accelerator_type=accelerator_type,
    endpoint_name=endpoint_name,
    wait=True
)

```

    Attaching to tmp1nu51x07_algo-1-1h732_1
    [36malgo-1-1h732_1  |[0m Requirement already satisfied: tqdm in /opt/conda/lib/python3.6/site-packages (from -r /opt/ml/model/code/requirements.txt (line 1)) (4.48.2)
    [36malgo-1-1h732_1  |[0m Requirement already satisfied: requests==2.22.0 in /opt/conda/lib/python3.6/site-packages (from -r /opt/ml/model/code/requirements.txt (line 2)) (2.22.0)
    [36malgo-1-1h732_1  |[0m Collecting regex
    [36malgo-1-1h732_1  |[0m   Downloading regex-2020.11.13-cp36-cp36m-manylinux2014_x86_64.whl (723 kB)
    [K     |████████████████████████████████| 723 kB 19.9 MB/s eta 0:00:01
    [36malgo-1-1h732_1  |[0m [?25hCollecting sentencepiece
    [36malgo-1-1h732_1  |[0m   Downloading sentencepiece-0.1.94-cp36-cp36m-manylinux2014_x86_64.whl (1.1 MB)
    [K     |████████████████████████████████| 1.1 MB 74.9 MB/s eta 0:00:01
    [36malgo-1-1h732_1  |[0m [?25hCollecting sacremoses
    [36malgo-1-1h732_1  |[0m   Downloading sacremoses-0.0.43.tar.gz (883 kB)
    [K     |████████████████████████████████| 883 kB 83.0 MB/s eta 0:00:01
    [36malgo-1-1h732_1  |[0m [?25hCollecting transformers==2.3.0
    [36malgo-1-1h732_1  |[0m   Downloading transformers-2.3.0-py3-none-any.whl (447 kB)
    [K     |████████████████████████████████| 447 kB 61.0 MB/s eta 0:00:01
    [36malgo-1-1h732_1  |[0m [?25hRequirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r /opt/ml/model/code/requirements.txt (line 2)) (1.25.10)
    [36malgo-1-1h732_1  |[0m Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r /opt/ml/model/code/requirements.txt (line 2)) (2020.6.20)
    [36malgo-1-1h732_1  |[0m Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r /opt/ml/model/code/requirements.txt (line 2)) (2.8)
    [36malgo-1-1h732_1  |[0m Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests==2.22.0->-r /opt/ml/model/code/requirements.txt (line 2)) (3.0.4)
    [36malgo-1-1h732_1  |[0m Requirement already satisfied: six in /opt/conda/lib/python3.6/site-packages (from sacremoses->-r /opt/ml/model/code/requirements.txt (line 5)) (1.15.0)
    [36malgo-1-1h732_1  |[0m Collecting click
    [36malgo-1-1h732_1  |[0m   Downloading click-7.1.2-py2.py3-none-any.whl (82 kB)
    [K     |████████████████████████████████| 82 kB 2.4 MB/s  eta 0:00:01
    [36malgo-1-1h732_1  |[0m [?25hRequirement already satisfied: joblib in /opt/conda/lib/python3.6/site-packages (from sacremoses->-r /opt/ml/model/code/requirements.txt (line 5)) (0.16.0)
    [36malgo-1-1h732_1  |[0m Collecting boto3
    [36malgo-1-1h732_1  |[0m   Downloading boto3-1.16.38-py2.py3-none-any.whl (130 kB)
    [K     |████████████████████████████████| 130 kB 83.7 MB/s eta 0:00:01
    [36malgo-1-1h732_1  |[0m [?25hRequirement already satisfied: numpy in /opt/conda/lib/python3.6/site-packages (from transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (1.19.1)
    [36malgo-1-1h732_1  |[0m Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /opt/conda/lib/python3.6/site-packages (from boto3->transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (0.10.0)
    [36malgo-1-1h732_1  |[0m Collecting botocore<1.20.0,>=1.19.38
    [36malgo-1-1h732_1  |[0m   Downloading botocore-1.19.38-py2.py3-none-any.whl (7.1 MB)
    [K     |████████████████████████████████| 7.1 MB 51.1 MB/s eta 0:00:01
    [36malgo-1-1h732_1  |[0m [?25hRequirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /opt/conda/lib/python3.6/site-packages (from boto3->transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (0.3.3)
    [36malgo-1-1h732_1  |[0m Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /opt/conda/lib/python3.6/site-packages (from botocore<1.20.0,>=1.19.38->boto3->transformers==2.3.0->-r /opt/ml/model/code/requirements.txt (line 6)) (2.8.1)
    [36malgo-1-1h732_1  |[0m Building wheels for collected packages: sacremoses
    [36malgo-1-1h732_1  |[0m   Building wheel for sacremoses (setup.py) ... [?25ldone
    [36malgo-1-1h732_1  |[0m [?25h  Created wheel for sacremoses: filename=sacremoses-0.0.43-py3-none-any.whl size=893259 sha256=237817276dc5546bc221ed43d290a8de93d7605e6f3aa53ac9769ec905fef79a
    [36malgo-1-1h732_1  |[0m   Stored in directory: /root/.cache/pip/wheels/49/25/98/cdea9c79b2d9a22ccc59540b1784b67f06b633378e97f58da2
    [36malgo-1-1h732_1  |[0m Successfully built sacremoses
    [36malgo-1-1h732_1  |[0m Installing collected packages: regex, sentencepiece, click, sacremoses, botocore, boto3, transformers
    [36malgo-1-1h732_1  |[0m   Attempting uninstall: botocore
    [36malgo-1-1h732_1  |[0m     Found existing installation: botocore 1.17.48
    [36malgo-1-1h732_1  |[0m     Uninstalling botocore-1.17.48:
    [36malgo-1-1h732_1  |[0m       Successfully uninstalled botocore-1.17.48
    [36malgo-1-1h732_1  |[0m [31mERROR: After October 2020 you may experience errors when installing or updating packages. This is because pip will change the way that it resolves dependency conflicts.
    [36malgo-1-1h732_1  |[0m 
    [36malgo-1-1h732_1  |[0m We recommend you use --use-feature=2020-resolver to test your packages with the new resolver before it becomes the default.
    [36malgo-1-1h732_1  |[0m 
    [36malgo-1-1h732_1  |[0m awscli 1.18.125 requires botocore==1.17.48, but you'll have botocore 1.19.38 which is incompatible.[0m
    [36malgo-1-1h732_1  |[0m Successfully installed boto3-1.16.38 botocore-1.19.38 click-7.1.2 regex-2020.11.13 sacremoses-0.0.43 sentencepiece-0.1.94 transformers-2.3.0
    [36malgo-1-1h732_1  |[0m Warning: Calling MMS with mxnet-model-server. Please move to multi-model-server.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,400 [INFO ] main com.amazonaws.ml.mms.ModelServer - 
    [36malgo-1-1h732_1  |[0m MMS Home: /opt/conda/lib/python3.6/site-packages
    [36malgo-1-1h732_1  |[0m Current directory: /
    [36malgo-1-1h732_1  |[0m Temp directory: /home/model-server/tmp
    [36malgo-1-1h732_1  |[0m Number of GPUs: 0
    [36malgo-1-1h732_1  |[0m Number of CPUs: 8
    [36malgo-1-1h732_1  |[0m Max heap size: 6972 M
    [36malgo-1-1h732_1  |[0m Python executable: /opt/conda/bin/python
    [36malgo-1-1h732_1  |[0m Config file: /etc/sagemaker-mms.properties
    [36malgo-1-1h732_1  |[0m Inference address: http://0.0.0.0:8080
    [36malgo-1-1h732_1  |[0m Management address: http://0.0.0.0:8080
    [36malgo-1-1h732_1  |[0m Model Store: /.sagemaker/mms/models
    [36malgo-1-1h732_1  |[0m Initial Models: ALL
    [36malgo-1-1h732_1  |[0m Log dir: /logs
    [36malgo-1-1h732_1  |[0m Metrics dir: /logs
    [36malgo-1-1h732_1  |[0m Netty threads: 0
    [36malgo-1-1h732_1  |[0m Netty client threads: 0
    [36malgo-1-1h732_1  |[0m Default workers per model: 8
    [36malgo-1-1h732_1  |[0m Blacklist Regex: N/A
    [36malgo-1-1h732_1  |[0m Maximum Response Size: 6553500
    [36malgo-1-1h732_1  |[0m Maximum Request Size: 6553500
    [36malgo-1-1h732_1  |[0m Preload model: false
    [36malgo-1-1h732_1  |[0m Prefer direct buffer: false
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,456 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerLifeCycle - attachIOStreams() threadName=W-9000-model
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,547 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - model_service_worker started with args: --sock-type unix --sock-name /home/model-server/tmp/.mms.sock.9000 --handler sagemaker_pytorch_serving_container.handler_service --model-path /.sagemaker/mms/models/model --model-name model --preload-model false --tmp-dir /home/model-server/tmp
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,548 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Listening on port: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,548 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - [PID] 51
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,549 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - MMS worker started.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,549 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Python runtime: 3.6.6
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,549 [INFO ] main com.amazonaws.ml.mms.wlm.ModelManager - Model model loaded.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,553 [INFO ] main com.amazonaws.ml.mms.ModelServer - Initialize Inference server with: EpollServerSocketChannel.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,561 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,561 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,561 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,561 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,561 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,561 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,561 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,561 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,613 [INFO ] main com.amazonaws.ml.mms.ModelServer - Inference API bind to: http://0.0.0.0:8080
    [36malgo-1-1h732_1  |[0m Model server started.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,619 [WARN ] pool-2-thread-1 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,623 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,624 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,624 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,624 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,624 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,625 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,626 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:04,627 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,339 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,346 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,365 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,369 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,371 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,377 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,394 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,425 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,804 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,804 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmp24wgl867
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,810 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,810 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmp0kvkwrol
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,828 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,828 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmpxfp1zhtj
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,839 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,839 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmp6kvg10dh
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,839 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,839 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmp56ik_0m_
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,842 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,843 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt not found in cache or force_download set to True, downloading to /home/model-server/tmp/tmphacdo8dr
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,848 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmp24wgl867 to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,849 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,850 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmp24wgl867
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,851 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,852 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmp0kvkwrol to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,853 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,853 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmp0kvkwrol
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,853 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,863 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,863 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,866 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmpxfp1zhtj to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,867 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,868 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmpxfp1zhtj
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,868 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,883 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmp56ik_0m_ to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,883 [INFO ] epollEventLoopGroup-4-7 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,884 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,885 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,885 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,886 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmp56ik_0m_
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,886 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,893 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmp6kvg10dh to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,895 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,895 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmp6kvg10dh
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,896 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,896 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,896 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,905 [INFO ] epollEventLoopGroup-4-1 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,905 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,905 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,911 [INFO ] epollEventLoopGroup-4-6 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,911 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - copying /home/model-server/tmp/tmphacdo8dr to cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,911 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,912 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,912 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - creating metadata file for /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,913 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - removing temp file /home/model-server/tmp/tmphacdo8dr
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,913 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,927 [INFO ] epollEventLoopGroup-4-3 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,927 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,928 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,933 [INFO ] epollEventLoopGroup-4-4 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,934 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,934 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,963 [INFO ] epollEventLoopGroup-4-8 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,963 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:05,963 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,056 [INFO ] epollEventLoopGroup-4-5 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,056 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,056 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,069 [INFO ] epollEventLoopGroup-4-2 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,070 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,070 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,492 [INFO ] W-9000-model ACCESS_LOG - /172.18.0.1:47050 "GET /ping HTTP/1.1" 200 8
    ![36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,885 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,887 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,906 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,907 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,912 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,913 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,928 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,929 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,934 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,935 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,964 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:06,965 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,057 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,059 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,070 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,073 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,458 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,534 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,581 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,591 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,591 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,669 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,769 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,774 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,921 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,922 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,990 [INFO ] epollEventLoopGroup-4-9 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,990 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:07,990 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,023 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,023 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,026 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,026 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,033 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,033 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,060 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,062 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,069 [INFO ] epollEventLoopGroup-4-11 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,069 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,070 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,083 [INFO ] epollEventLoopGroup-4-10 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,083 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,084 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,089 [INFO ] epollEventLoopGroup-4-12 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,089 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,089 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,102 [INFO ] epollEventLoopGroup-4-13 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,102 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,102 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539 in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,132 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,132 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,167 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,167 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,178 [INFO ] epollEventLoopGroup-4-14 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,179 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,179 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,204 [INFO ] epollEventLoopGroup-4-15 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,204 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,204 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,207 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,207 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,243 [INFO ] epollEventLoopGroup-4-16 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,243 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,243 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f in 1 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,991 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:08,992 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,070 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,072 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,084 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,085 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,089 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,090 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,102 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,104 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,179 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,181 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,204 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,207 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,244 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,248 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,516 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,706 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,714 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,741 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,825 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,871 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,911 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,981 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,982 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:09,983 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,047 [INFO ] epollEventLoopGroup-4-3 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,048 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,049 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf in 2 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,181 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,182 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,185 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,185 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,221 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,221 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,225 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,226 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,233 [INFO ] epollEventLoopGroup-4-2 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,233 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,233 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b in 2 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,239 [INFO ] epollEventLoopGroup-4-6 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,240 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,240 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e in 2 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,262 [INFO ] epollEventLoopGroup-4-1 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,262 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,263 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22 in 2 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,283 [INFO ] epollEventLoopGroup-4-8 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,283 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,283 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a in 2 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,315 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,315 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,317 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,317 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,350 [INFO ] epollEventLoopGroup-4-4 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,351 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,351 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d in 2 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,353 [INFO ] epollEventLoopGroup-4-7 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,353 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,353 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40 in 2 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,362 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,363 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,397 [INFO ] epollEventLoopGroup-4-5 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,397 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:10,397 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41 in 2 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,049 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,050 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,234 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,235 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,240 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,241 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,263 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,264 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,283 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,285 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,351 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,353 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,354 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,355 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,397 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,401 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,542 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,853 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,914 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,928 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:12,990 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,012 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,013 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,065 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,085 [INFO ] epollEventLoopGroup-4-9 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,085 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,085 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3 in 3 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,091 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,119 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,293 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,294 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,336 [INFO ] epollEventLoopGroup-4-12 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,337 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,337 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed in 3 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,338 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,338 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,397 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,397 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,411 [INFO ] epollEventLoopGroup-4-10 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,411 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,412 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a in 3 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,441 [INFO ] epollEventLoopGroup-4-11 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,441 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,441 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de in 3 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,457 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,457 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,512 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,512 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,513 [INFO ] epollEventLoopGroup-4-13 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,513 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,513 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c in 3 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,516 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,516 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,555 [INFO ] epollEventLoopGroup-4-14 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,556 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,556 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130 in 3 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,562 [INFO ] epollEventLoopGroup-4-15 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,563 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,563 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321 in 3 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,582 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,582 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,625 [INFO ] epollEventLoopGroup-4-16 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,625 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:13,625 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2 in 3 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,085 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,087 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,337 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,338 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,412 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,413 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,442 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,443 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,498 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,513 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,514 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,556 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,558 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,563 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,565 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,625 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,627 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,897 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,907 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,908 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,981 [INFO ] epollEventLoopGroup-4-3 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,982 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,983 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363 in 5 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:16,992 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,109 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,176 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,202 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,241 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,296 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,336 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,337 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,407 [INFO ] epollEventLoopGroup-4-8 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,407 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,408 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86 in 5 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,456 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,456 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,507 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,507 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,513 [INFO ] epollEventLoopGroup-4-6 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,513 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,514 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad in 5 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,563 [INFO ] epollEventLoopGroup-4-5 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,564 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,564 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735 in 5 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,583 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,583 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,584 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,584 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,606 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,606 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,619 [INFO ] epollEventLoopGroup-4-1 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,619 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,620 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f in 5 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,631 [INFO ] epollEventLoopGroup-4-2 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,631 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,631 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee in 5 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,642 [INFO ] epollEventLoopGroup-4-7 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,642 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,642 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf in 5 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,708 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,708 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,742 [INFO ] epollEventLoopGroup-4-4 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,742 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:17,742 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139 in 5 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:21,983 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:21,984 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,405 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,408 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,409 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,514 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,515 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,564 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,566 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,620 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,622 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,631 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,633 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,643 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,645 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,742 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,745 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,762 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,763 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,955 [INFO ] epollEventLoopGroup-4-9 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,955 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:22,956 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e in 8 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,069 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,267 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,276 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,359 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,423 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,469 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,554 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,555 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,587 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,591 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,592 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,597 [INFO ] epollEventLoopGroup-4-10 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,598 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,598 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404 in 8 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,655 [INFO ] epollEventLoopGroup-4-13 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,655 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,655 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345 in 8 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,704 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,704 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,746 [INFO ] epollEventLoopGroup-4-11 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,746 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,746 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0 in 8 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,812 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,812 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,854 [INFO ] epollEventLoopGroup-4-12 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,855 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,855 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0 in 8 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,859 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,859 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,865 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,865 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,902 [INFO ] epollEventLoopGroup-4-14 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,902 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,902 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449 in 8 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,908 [INFO ] epollEventLoopGroup-4-15 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,908 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,908 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff in 8 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,921 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,921 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,963 [INFO ] epollEventLoopGroup-4-16 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,963 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:23,963 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613 in 8 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:30,956 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:30,957 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,374 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,598 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,599 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,656 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,657 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,664 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,664 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,705 [INFO ] epollEventLoopGroup-4-3 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,705 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,705 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c in 13 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,747 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,748 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,855 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,856 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,902 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,904 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,909 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,910 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,963 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:31,965 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,070 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,225 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,251 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,396 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,397 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,438 [INFO ] epollEventLoopGroup-4-8 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,438 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,439 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492 in 13 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,487 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,582 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,602 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,620 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,694 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,694 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,709 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,709 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,763 [INFO ] epollEventLoopGroup-4-6 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,764 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,764 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0 in 13 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,780 [INFO ] epollEventLoopGroup-4-2 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,780 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,780 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903 in 13 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,840 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,840 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,883 [INFO ] epollEventLoopGroup-4-1 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,883 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,883 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192 in 13 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,902 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,902 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,930 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,930 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,931 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,931 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,943 [INFO ] epollEventLoopGroup-4-4 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,943 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,944 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4 in 13 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,971 [INFO ] epollEventLoopGroup-4-7 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,971 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,971 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93 in 13 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,978 [INFO ] epollEventLoopGroup-4-5 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,978 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:32,978 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d in 13 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:44,706 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:44,707 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,109 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,390 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,390 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,430 [INFO ] epollEventLoopGroup-4-9 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,430 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,430 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970 in 21 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,439 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,440 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,764 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,765 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,781 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,781 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,841 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,883 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,884 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,944 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,945 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,972 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,973 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,978 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:45,979 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,219 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,220 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,275 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,279 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,283 [INFO ] epollEventLoopGroup-4-10 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,284 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,284 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b in 21 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,465 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,502 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,572 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,572 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,610 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,610 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,613 [INFO ] epollEventLoopGroup-4-12 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,613 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,613 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36 in 21 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,704 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,704 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,763 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,763 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,766 [INFO ] epollEventLoopGroup-4-11 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,767 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,767 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764 in 21 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,805 [INFO ] epollEventLoopGroup-4-15 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,806 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,806 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a in 21 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,870 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,870 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,912 [INFO ] epollEventLoopGroup-4-13 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,912 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,912 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731 in 21 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,924 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,924 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,929 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,929 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,965 [INFO ] epollEventLoopGroup-4-16 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,965 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,965 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002 in 21 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,972 [INFO ] epollEventLoopGroup-4-14 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,972 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:53:46,972 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46 in 21 seconds.



```python
predictor.delete_endpoint()
```

    Gracefully stopping... (press Ctrl+C again to force)



```python
result = predictor.predict("Somebody just left - guess who.")
print(np.argmax(result, axis=1))
```

    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:04,616 [WARN ] pool-2-thread-1 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:06,430 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:06,431 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:06,845 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,135 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,135 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,176 [INFO ] epollEventLoopGroup-4-3 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,176 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,176 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051 in 34 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,285 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,286 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,614 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,614 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,695 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,767 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,768 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,806 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,807 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,912 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,914 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,966 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,967 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,972 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,973 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,990 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:07,991 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,040 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,055 [INFO ] epollEventLoopGroup-4-8 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,056 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,056 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123 in 34 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,216 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,345 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,345 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,388 [INFO ] epollEventLoopGroup-4-6 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,388 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,388 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79 in 34 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,419 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,516 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,516 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,558 [INFO ] epollEventLoopGroup-4-2 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,558 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,558 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a in 34 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,561 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,563 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,597 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,781 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,781 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,823 [INFO ] epollEventLoopGroup-4-1 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,823 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,823 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a in 34 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,851 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,851 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,855 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,855 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,893 [INFO ] epollEventLoopGroup-4-5 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,893 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,893 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b in 34 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,899 [INFO ] epollEventLoopGroup-4-7 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,900 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,900 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd in 34 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,909 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,909 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,950 [INFO ] epollEventLoopGroup-4-4 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,950 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:08,950 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab in 34 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:41,176 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:41,177 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:41,586 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:41,892 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:41,892 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:41,932 [INFO ] epollEventLoopGroup-4-9 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051-88efcede Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:41,933 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:41,933 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051-88efcede in 55 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,056 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,057 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,388 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,389 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,466 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,559 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,560 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,775 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,775 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,800 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,817 [INFO ] epollEventLoopGroup-4-10 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123-5de0f6fa Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,817 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,818 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123-5de0f6fa in 55 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,823 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,824 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,893 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,894 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,900 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,901 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,951 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,952 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:42,990 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,164 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,164 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,235 [INFO ] epollEventLoopGroup-4-11 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79-e9fa0865 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,235 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,235 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79-e9fa0865 in 55 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,301 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,301 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,365 [INFO ] epollEventLoopGroup-4-12 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a-58c42159 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,365 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,365 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a-58c42159 in 55 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,378 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,427 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,475 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,475 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,662 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,662 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,706 [INFO ] epollEventLoopGroup-4-15 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd-3b0cc419 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,706 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,706 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd-3b0cc419 in 55 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,726 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,726 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,766 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,766 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,767 [INFO ] epollEventLoopGroup-4-13 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a-3bf7224c Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,767 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,767 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a-3bf7224c in 55 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,776 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,776 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,808 [INFO ] epollEventLoopGroup-4-16 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab-6c0c94ed Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,808 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,808 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab-6c0c94ed in 55 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,817 [INFO ] epollEventLoopGroup-4-14 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b-46e0446a Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,817 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:54:43,818 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b-46e0446a in 55 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:04,616 [WARN ] pool-2-thread-2 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:36,933 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:36,934 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:37,346 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:37,646 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:37,646 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:37,686 [INFO ] epollEventLoopGroup-4-3 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051-88efcede-7f7375d4 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:37,686 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:37,686 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051-88efcede-7f7375d4 in 89 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:37,818 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:37,819 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,232 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,236 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,237 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,366 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,367 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,523 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,523 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,564 [INFO ] epollEventLoopGroup-4-8 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123-5de0f6fa-6dd0dd5f Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,564 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,564 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123-5de0f6fa-6dd0dd5f in 89 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,653 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,706 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,707 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,768 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,769 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,780 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,809 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,810 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,818 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,819 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,962 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:38,963 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,030 [INFO ] epollEventLoopGroup-4-6 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79-e9fa0865-bccecb42 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,031 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,031 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79-e9fa0865-bccecb42 in 89 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,116 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,116 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,158 [INFO ] epollEventLoopGroup-4-2 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a-58c42159-894028e4 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,158 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,158 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a-58c42159-894028e4 in 89 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,272 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,350 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,375 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,390 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,558 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,558 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,599 [INFO ] epollEventLoopGroup-4-1 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd-3b0cc419-ba917b90 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,599 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,599 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd-3b0cc419-ba917b90 in 89 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,650 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,650 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,662 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,662 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,683 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,683 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,691 [INFO ] epollEventLoopGroup-4-4 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a-3bf7224c-b5f22dd5 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,691 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,691 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a-3bf7224c-b5f22dd5 in 89 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,703 [INFO ] epollEventLoopGroup-4-5 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b-46e0446a-4ac13267 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,703 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,703 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b-46e0446a-4ac13267 in 89 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,726 [INFO ] epollEventLoopGroup-4-7 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab-6c0c94ed-119bde2e Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,726 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:55:39,726 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab-6c0c94ed-119bde2e in 89 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:56:04,616 [WARN ] pool-2-thread-1 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:04,616 [WARN ] pool-2-thread-1 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:06,687 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:06,688 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:07,089 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:07,383 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:07,384 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:07,424 [INFO ] epollEventLoopGroup-4-9 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051-88efcede-7f7375d4-8a787685 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:07,424 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:07,424 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051-88efcede-7f7375d4-8a787685 in 144 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:07,565 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:07,566 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:07,967 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,031 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,032 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,159 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,160 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,275 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,275 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,316 [INFO ] epollEventLoopGroup-4-10 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123-5de0f6fa-6dd0dd5f-d364e90c Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,316 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,316 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123-5de0f6fa-6dd0dd5f-d364e90c in 144 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,438 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,577 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,599 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,600 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,691 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,693 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,704 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,705 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,726 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,729 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,729 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,729 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,792 [INFO ] epollEventLoopGroup-4-11 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79-e9fa0865-bccecb42-519da666 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,792 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,792 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79-e9fa0865-bccecb42-519da666 in 144 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,872 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,872 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,914 [INFO ] epollEventLoopGroup-4-12 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a-58c42159-894028e4-05dbddc8 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,914 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:08,914 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a-58c42159-894028e4-05dbddc8 in 144 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,039 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,133 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,214 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,247 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,327 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,327 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,369 [INFO ] epollEventLoopGroup-4-13 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd-3b0cc419-ba917b90-3b6e90db Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,369 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,369 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd-3b0cc419-ba917b90-3b6e90db in 144 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,463 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,463 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,496 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,496 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,517 [INFO ] epollEventLoopGroup-4-14 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a-3bf7224c-b5f22dd5-649b6c97 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,517 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,517 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a-3bf7224c-b5f22dd5-649b6c97 in 144 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,537 [INFO ] epollEventLoopGroup-4-15 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b-46e0446a-4ac13267-680f1405 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,537 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,537 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b-46e0446a-4ac13267-680f1405 in 144 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,544 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,544 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,583 [INFO ] epollEventLoopGroup-4-16 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab-6c0c94ed-119bde2e-a31f0082 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,583 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:57:09,583 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab-6c0c94ed-119bde2e-a31f0082 in 144 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:58:04,616 [WARN ] pool-2-thread-2 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:04,616 [WARN ] pool-2-thread-2 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:31,424 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:31,425 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:31,826 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,119 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,119 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,159 [INFO ] epollEventLoopGroup-4-3 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051-88efcede-7f7375d4-8a787685-9013847b Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,159 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,159 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-da4e39e7-7d78fac3-0ead0edf-bc76efb3-9863c363-5004a66e-e78b0f0c-bb745970-e8040051-88efcede-7f7375d4-8a787685-9013847b in 233 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,316 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,317 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,725 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,793 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,794 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,915 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:32,916 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,021 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,021 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,062 [INFO ] epollEventLoopGroup-4-8 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123-5de0f6fa-6dd0dd5f-d364e90c-2894d81a Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,062 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,062 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-c5be780f-32317539-45f1fd22-a6406eed-344f6c86-2663b404-84f59492-f143d24b-ce93e123-5de0f6fa-6dd0dd5f-d364e90c-2894d81a in 233 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,199 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,320 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,369 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,370 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,505 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,506 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,518 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,519 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,538 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,540 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,550 [INFO ] epollEventLoopGroup-4-6 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79-e9fa0865-bccecb42-519da666-d5415b34 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,550 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,550 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-7e307ee3-ec557785-43921c4b-946f219a-a93c8cad-0ef700d0-296a8903-72e75b36-36fcef79-e9fa0865-bccecb42-519da666-d5415b34 in 233 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,584 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.mms.sock.9000
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,584 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Connection accepted: /home/model-server/tmp/.mms.sock.9000.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,626 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,626 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,692 [INFO ] epollEventLoopGroup-4-2 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a-58c42159-894028e4-05dbddc8-caa9d3c2 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,692 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,692 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-3aabe6f8-d8974b9e-0b117a9a-cd22115c-fd2aa16f-5c7ce345-9644e9c0-65578764-0b3e4f4a-58c42159-894028e4-05dbddc8-caa9d3c2 in 233 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,780 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,940 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:33,989 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,002 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - PyTorch version 1.3.1 available.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,076 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,077 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,132 [INFO ] epollEventLoopGroup-4-1 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd-3b0cc419-ba917b90-3b6e90db-c655a1ec Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,132 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,132 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b5d157e8-e6b3269b-62ebfd0d-e237b130-cdb77139-96f94613-577b659d-e2ec3002-6c4e52dd-3b0cc419-ba917b90-3b6e90db-c655a1ec in 233 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,252 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,252 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,277 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,277 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,293 [INFO ] epollEventLoopGroup-4-4 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a-3bf7224c-b5f22dd5-649b6c97-baed3d75 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,293 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,293 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b3b098d6-16d86b3b-cbdcdb40-6170b321-546909bf-fb0a99ff-afc44e93-d1ad2a3a-bede382a-3bf7224c-b5f22dd5-649b6c97-baed3d75 in 233 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,294 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - Loading BERT tokenizer...
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,294 [INFO ] W-9000-model-stdout com.amazonaws.ml.mms.wlm.WorkerLifeCycle - loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,318 [INFO ] epollEventLoopGroup-4-7 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b-46e0446a-4ac13267-680f1405-82f4932c Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,318 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,319 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-579a70d5-ff243b34-43b1414e-cba7c6de-a04ea8ee-8095b449-5467fba4-41833b46-353b597b-46e0446a-4ac13267-680f1405-82f4932c in 233 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,335 [INFO ] epollEventLoopGroup-4-5 com.amazonaws.ml.mms.wlm.WorkerThread - 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab-6c0c94ed-119bde2e-a31f0082-82b22a05 Worker disconnected. WORKER_STARTED
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,335 [WARN ] W-9000-model com.amazonaws.ml.mms.wlm.BatchAggregator - Load model failed: model, error: Worker died.
    [36malgo-1-1h732_1  |[0m 2020-12-17 09:59:34,335 [INFO ] W-9000-model com.amazonaws.ml.mms.wlm.WorkerThread - Retry worker: 9000-b8319d76-de42490f-f3a9be41-738cd6f2-0769a735-b59db7a0-a6c8f192-043c1731-e3dad5ab-6c0c94ed-119bde2e-a31f0082-82b22a05 in 233 seconds.
    [36malgo-1-1h732_1  |[0m 2020-12-17 10:00:04,616 [WARN ] pool-2-thread-1 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.
    [36malgo-1-1h732_1  |[0m 2020-12-17 10:01:04,616 [WARN ] pool-2-thread-1 com.amazonaws.ml.mms.metrics.MetricCollector - worker pid is not available yet.



```python

```
