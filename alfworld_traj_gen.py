import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
import yaml
import alfworld
import alfworld.agents.environment.alfred_tw_env
import sys
import json
import uuid

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "AronXiang/RetrospexLLaMA3"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f'DEVICE = {DEVICE}')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enables 4-bit quantization
    bnb_4bit_quant_type="nf4",  # NormalFloat4 (NF4) for better precision
    bnb_4bit_compute_dtype=torch.float16,  # Compute in float16 (reduce memory usage)
    bnb_4bit_use_double_quant=True  # Double quantization for additional compression
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)


model.to(DEVICE)

tokenizer.eos_token = "\n"
tokenizer.pad_token_id = tokenizer.eos_token_id  # Prevents warning
model.config.pad_token_id = tokenizer.eos_token_id
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer,eos_token_id=tokenizer.eos_token_id,  # Stop generation at end-of-sequence token
        pad_token_id=tokenizer.pad_token_id, 
                        max_new_tokens = 200, device_map="auto")



with open('base_config.yaml') as reader:
    config = yaml.safe_load(reader)

split = "eval_out_of_distribution"
# config['env']['type'] = 'AlfredTwEnv'
# print(config['env']['type'])
env = getattr(alfworld.agents.environment.alfred_tw_env, 'AlfredTWEnv')(config, train_eval=split)
env = env.init_env(batch_size=1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

def alfworld_run(conversation_history, to_print=False, ob=''):
    prompt = ''
    
    if to_print:
        sys.stdout.flush()
    
    for i in range(1, 30):
        llm_response = llm_pipeline(conversation_history, pad_token_id=llm_pipeline.tokenizer.eos_token_id)[0]['generated_text']
        llm_response = llm_response[-1]['content']

        action =  llm_response.split("ACTION:")[-1].strip() if "ACTION:" in llm_response else llm_response.strip()

        # print(f'llm_response = {llm_response[-1]}')
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        
        if action.startswith('THOUGHT:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {llm_response}\nObs {i}: {observation}, Reward = {reward}')
            print('.'*10)
            sys.stdout.flush()
        
        
        conversation_history.append({'role':'assistant', 'content':llm_response, 'reward':reward})
        admissible_commands = ", ".join(info['admissible_commands'][0])
        conversation_history.append({'role':'user', 'content':observation + "\n Here is the updated list of the available actions provided to you: " +admissible_commands, 'reward':None})
        if done:
            return reward, conversation_history
    return 0, conversation_history

system_prompt = f'''Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. 
At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 

For each of your turn, you will be given a list of actions which you can choose one to perform in this turn.

You can choose either of given option: \"THOUGHT\" or \"ACTION\". If you choose \"THOUGHT\", you should first think about the current condition and plan for your future actions, 
and then output your action in this turn. 
Your output must strictly follow this format:\"THOUGHT: your thoughts.\nACTION: your next action\n\"; 

If you choose \"ACTION\", you should directly output the action in this turn. Your output must strictly follow this format:\"ACTION: your next action\n\".
 
 After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. 
 if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.
 \n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as **illegal**. \n2.Think when necessary, try to act directly more in the process.
'''



prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

cnts = [0] * 6
rs = [0] * 6

conversations_traj = []

# first 0 to 9 
# second 10 to 19
for _ in range(20,100):
    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    alfworld_id = str(uuid.uuid4())
    
    conversation_history = []
    conversation_history.append({"role":"user", "content":system_prompt})
    response = llm_pipeline(conversation_history, pad_token_id=llm_pipeline.tokenizer.eos_token_id)  #calling pipleline
    assistant_response = response[0]['generated_text']
    conversation_history.append({'role':"assistant", 'content': assistant_response[-1]['content']})
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            # prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0'] + '\nHere is the task.\n'
            # prompt = conversation_history.append({'role':'user', 'content':'Here is your task.'+ob})
            # print('admissible_commands', info['admissible_commands'])
            admissible_commands = ", ".join(info['admissible_commands'][0])
            conversation_history.append({'role':'user', 'content':'Here is your task.'+ob + "\n Given are the the available actions provided to you: " +admissible_commands})
            print(k, v)
            # print(print(f'Observation = {ob}'))
            r, conversation_history = alfworld_run(conversation_history, ob=ob)
            rs[i] += r
            cnts[i] += 1
    # print(_+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
    conversations_traj.append({
        'traj_id': alfworld_id,
        'conversations': conversation_history
    })
    print('------------------------------------------------\n')


json_file_path = "alfworld_conversations_20_99.json"
with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(conversations_traj, f, indent=2, ensure_ascii=False)

# with open("alfworld_conversations.json", "r", encoding="utf-8") as json_file:
#     loaded_data = json.load(json_file)

# print("Loaded Conversation History:", json.dumps(loaded_data, indent=2, ensure_ascii=False))