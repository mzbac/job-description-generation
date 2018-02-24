# Job description generation 

# Getting started
```
git clone https://github.com/mzbac/job-description-generation.git
```
# cloud setting
- select Deep Learning AMI (Ubuntu) Version 4.0 - ami-b40f8bcc
- ssh to remote ec2
- source activate pytorch_p36

# Serving model 
```
cd host
export FLASK_APP=predictions.py
flask run
```
# Requirements 
- python 3
- pytorch
