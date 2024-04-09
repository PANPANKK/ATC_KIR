This paper presents a small sample-driven keyword information extraction framework, SLKIR, for the air traffic control domain.

The main contributions of this paper are as follows:
   
    We conducted an in-depth study on constructing a robust KIR model driven by small sample data. Specifically, we built an end-to-end KIR deep 
    learning framework based on the MHLA mechanism. 
    
    We cleverly designed a discriminative task based on prompt information. By discriminating whether the control instructions contain externally
    inputted prompt information, we enhance the semantic understanding of the input control instructions by the backbone network. 
   
    We proposed a loss function optimization strategy for addressing the issue of boundary word information sparsity in the KIR process. This 
    optimization strategy can enhance the model’s learning capability. 
    
    Building on previous research10, 11, and considering the characteristics of air traffic control as well as the key information required by 
    intelligent agents to execute instructions, this paper provides a detailed classification of key information categories in control instructions, 
    including Callsign, Action, Action Value, and Condition.

The data was annotated using the Doccano platform, with the key information of air traffic control instructions categorized into four types: Callsign, 
Action, Action Value, and Condition.

The format of the annotated data is as follows:
1.{"id":512,"text":"东方两八洞六联系地面幺两幺点六再见","label":[[0,6,"呼号"],[6,8,"动作"],[8,15,"动作值"]],"Comments":[]}
2.{"id":513,"text":"三五右跑道外等吉祥幺幺拐六","label":[[0,6,"前提条件"],[6,7,"动作"],[7,13,"呼号"]],"Comments":[]}

This paper introduces a prompt classification task, hence requiring the annotated data to be transformed into the following format: 
1.
{"content": "东方两八洞六联系地面幺两幺点六再见", "result_list": [{"text": "东方两八洞六", "start": 0, "end": 6}], "prompt": "呼号"}
{"content": "东方两八洞六联系地面幺两幺点六再见", "result_list": [{"text": "地面幺两幺点六", "start": 8, "end": 15}], "prompt": "动作值"}
{"content": "东方两八洞六联系地面幺两幺点六再见", "result_list": [{"text": "联系", "start": 6, "end": 8}], "prompt": "动作"}
{"content": "东方两八洞六联系地面幺两幺点六再见", "result_list": [], "prompt": "前提条件"}
2.
{"content": "三五右跑道外等吉祥幺幺拐六", "result_list": [], "prompt": "动作值"}
{"content": "三五右跑道外等吉祥幺幺拐六", "result_list": [{"text": "三五右跑道外", "start": 0, "end": 6}], "prompt": "前提条件"}
{"content": "三五右跑道外等吉祥幺幺拐六", "result_list": [{"text": "等", "start": 6, "end": 7}], "prompt": "动作"}
{"content": "三五右跑道外等吉祥幺幺拐六", "result_list": [{"text": "吉祥幺幺拐六", "start": 7, "end": 13}], "prompt": "呼号"}

When the value of result_list is empty, the sample is a negative example for the prompt classification task, and its classification task label is 
marked as 0. When the value of result_list is not empty, the sample is a positive example for the prompt classification task, and its classification
task label is marked as 1. Considering the imbalance between positive and negative sample classes, the prompt classification task during the training
process employs a weighted Binary Cross-Entropy (BCE) loss, the parameters of which can be found in the paper.

Execution Steps:
1.Prepare the data and place it under the './data/cner/final_data/' folder.
2.Navigate to the current directory and execute the training code with the following command:
python train.py 
    --train_path "./data/cner/final_data/XXX.txt" \
    --dev_path "./data/cner/final_data/XXX.txt" \
    --save_dir "./checkpoint" \
    --learning_rate 1e-5 \
    --batch_size 64 \
    --max_seq_len 60 \
    --num_epochs 200 \
    --model "uie_base_pytorch" \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device "gpu"
3.Load the saved code, prepare the test data, and execute the following test code:
python evaluate.py
    --model_path "./checkpoint/model_best" \
    --test_path "./data/cner/final_data/XXX.txt" \
    --batch_size 64 \
    --max_seq_len 60
    
Pre-trained parameters：  
You can find the pre-trained parameters in this link "https://drive.google.com/drive/folders/1HPOkcWM0nfOelpIv2J12v1kBvbbO5olp?usp=drive_link".

Data Declaration:
The small-sample data used for training in this article, due to domain restrictions, cannot be publicly disclosed here. Interested researchers 
may obtain it through a reasonable request; please contact: darcy981020@gmail.com. The test dataset consists of training flight data from a specific
location, accessible in the corresponding data folder. Detailed data descriptions can be found in the paper; please cite the source before use.

Reference Note:
This code is based on the official UIE code provided by PaddlePaddle, with a series of innovative improvements made for the features of the air 
traffic control dataset.



