from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import yaml
import alfworld
import alfworld.agents.environment
import alfworld.agents.environment.alfred_tw_env


from inference import cal_prob, MappingValidActionList, get_llama3_8b_response_multi
from alfworld_iql import ImplicitQLearning
from sentence_transformers import SentenceTransformer

args = {
    'env_step_limit': 15,
    'round': 1,
    'task_id': 5,
    'beams': 4,
    'seed': 42,
    'output_file': 'alfworld_result.txt',
    'spm_path': 'unigram_8k.model',
    'rom_path': 'zork1.z5',
    'discount': 0.99,
    'hidden_dim': 128,
    'embedding_dim': 64,
    'n_hidden': 2,
    'alpha': 0.005,
    'tau': 0.7,
    'beta': 3.0,
    'learning_rate': 0.0001,
    'discount_prob': 0.95,
    'limit_prob': 0.6,
    'iql_path': 'final_iql_alfworld.pt',
    'llm_path': None
}
from types import SimpleNamespace
args = SimpleNamespace(**args)



def load_model_iql(path, args):
    iql = ImplicitQLearning(
        args,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate)
    )
    iql.load_state_dict(torch.load(path))
    return iql

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def normalize(lst):
    max_v = max(lst)
    min_v = min(lst)
    if max_v != min_v:
        result = [(i-min_v)/(max_v-min_v) for i in lst]
    else:
        result = [0.5 for i in range(len(lst))]
    return result

def decide_action(probs, q_values, step, args):
    # normalize 2 list
    # probs = torch.exp(probs)
    # probs = list(probs)
    assert len(probs) == len(q_values)
    nor_prob = normalize(probs)
    nor_q = normalize(q_values)
    print(nor_prob)
    print(nor_q)
    lam = args.discount_prob**step
    lam = max(args.limit_prob, lam)
    score = [nor_prob[i]*lam+nor_q[i]*(1-lam) for i in range(len(nor_q))]
    max_score = max(score)
    return score.index(max_score)


def generate(model, tokenizer, messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    if input_ids.shape[1] >= 4048:
        messages = messages[-10:]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    print(result)
    return result

def filter(response):
    return response.split('ACTION:')[-1].strip()

def alfworld_run(iql, sbert, args, info, model, tokenizer,env, prompt, to_print=True, ob=''):
    last_ob = 'Start Now!'
    des = ob
    for i in range(1, 30):
        valid_actions = list(info['admissible_commands'])[0]
        print('-------valid: ',valid_actions)
        # response = generate(model,tokenizer, prompt).strip()
        responses = get_llama3_8b_response_multi(model, tokenizer, prompt, args.beams)
        actions = [filter(response) for response in responses]
        print('-------actions: ',actions)
        actions = MappingValidActionList(actions, valid_actions, sbert, args.beams)
        print('-------mapped: ',actions)
        q_values = iql.get_q(des, last_ob, actions)
        print('-------q_values: ',q_values)
        prob_scores = []
        for a in actions:
            p = cal_prob(a, prompt, model, tokenizer)
            # p = 0
            prob_scores.append(float(p))
        print('-------probs: ',prob_scores)
        action = actions[decide_action(prob_scores, q_values, i, args)]
        print('-------action: ',action)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        prompt.append({'role':'assistant','content':'ACTION: '+action})
        prompt.append({'role':'user','content':observation})
        last_ob = observation
        if done:
            return reward
    return 0

def main():
    # args = parse_args()

    # model_id = args.llm_path
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_id,device_map='auto',torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    iql = load_model_iql(args.iql_path, args)
    # sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    
    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    
    split = "eval_out_of_distribution"

    # env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = getattr(alfworld.agents.environment.alfred_tw_env, 'AlfredTWEnv')(config, train_eval=split)
    env = env.init_env(batch_size=1)

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

    for _ in range(10):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(name)
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                messages = [
                    {'role':'system','content':'You are a helpful, respectful and honest assistant.'},
                    {'role':'user','content':'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: \"THOUGHT\" or \"ACTION\". If you choose \"THOUGHT\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THOUGHT: your thoughts.\n ACTION: your next action\n\"; If you choose \"ACTION\", you should directly output the action in this turn. Your output must strictly follow this format:\"ACTION: your next action\n\". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.'},
                    {'role':'assistant','content':"OK. I'll follow your instructions and try my best to solve the task."},
                    {'role':'user','content':'Here is your task. '+ob}
                ]
                r = alfworld_run(iql, sbert, args, info, model, tokenizer, env, messages, ob=ob)
                rs[i] += r
                cnts[i] += 1
                break
        print(_+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
        print('------------\n')
        with open(args.output_file,'a') as f:
            f.write(str(r))
            f.write('\n')

main()
