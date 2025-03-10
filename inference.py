import os

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import gc


def get_llama3_8b_response(model, tokenizer, messages):
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

def get_llama3_8b_response_multi(model, tokenizer, messages, k):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    print(input_ids.shape)
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
            temperature=0.9,
            top_p=0.95,
            num_return_sequences=k
        )
    # print(outputs.keys())
    # print(outputs['scores'])
    responses = []
    for i in range(k):
        response = outputs[i][input_ids.shape[-1]:]
        result = tokenizer.decode(response, skip_special_tokens=True)
        responses.append(result)
    # responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(result)
    gc.collect()
    torch.cuda.empty_cache()
    return responses

def cal_prob(target_text,messages,model,tokenizer):
    temp_messages = messages[:]
    temp_messages.append({'role':'assistant','content':target_text})
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    encodings = tokenizer.apply_chat_template(temp_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # print(input_ids.shape)
    # print(input_ids)
    # print(encodings.shape)
    # print(encodings)

    labels = encodings.clone()
    labels[:,:input_ids.shape[1]] = -100
    # print(labels)
    # print(labels)

    with torch.no_grad():
        outputs = model.forward(encodings, labels=labels)
        loss = outputs[0]
        text_prob=torch.exp(-loss)#**(len(target_text))
    return text_prob

def get_actions_origin(predictions):
    action_lst = []
    for pred in predictions:
        pred_action = pred.replace("\n",'').split('Action:')[-1]
        action_lst.append(pred_action.lower())
    return action_lst

def MappingValidActionList(predictions, validActions, sbert_model, k):
    action_lst = []
    remain = []
    if validActions == []:
        for pred in predictions[:k]:
            pred_action = pred.replace("\n",'').split('Action:')[-1]
            action_lst.append(pred_action)
        return action_lst

    if len(validActions) < k:
        return validActions

    for pred in predictions[:k]:
        pred_action = pred.replace("\n",'').split('Action:')[-1]
        if pred_action.lower() in validActions:
            if pred_action.lower() not in action_lst:
                action_lst.append(pred_action.lower())
        else:
            if pred_action.lower() not in remain:
                remain.append(pred_action.lower())
    lr = len(remain)
    print(remain)
    if lr != 0 and sbert_model:
        # text = remain+validActions
        # embeddings = sbert_model.encode(text, show_progress_bar=False)
        # for i in range(lr):
        #     similarity = cosine_similarity([embeddings[i]], embeddings[lr:])
        #     s = similarity.tolist()[0]
        #     sorted_id = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
        #     for j in sorted_id:
        #         if validActions[j] not in action_lst:
        #             action_lst.append(validActions[j])
        #             break

        pred_vectors = sbert_model.encode(predictions[:k], batch_size=k, show_progress_bar=False)
        valid_action_vectors = sbert_model.encode(validActions, batch_size=min(len(validActions), 128), show_progress_bar=False)
        similarity_matrix = cosine_similarity(pred_vectors, valid_action_vectors)
        sum_similarities = similarity_matrix.sum(axis=0)
        top_indices = np.argpartition(sum_similarities, -k)[-k:]
        count = 0
        for i in range(k):
            action = validActions[top_indices[i]]
            if action not in action_lst:
                action_lst.append(action)
                count += 1
            if count == lr:
                break
    # if lr == 0 and len(action_lst)<k:
    #     choose_set = set(validActions)-set(action_lst)
    #     action_lst += random.choices(list[choose_set], k=k-len(action_lst))
    gc.collect()
    torch.cuda.empty_cache()
    return action_lst