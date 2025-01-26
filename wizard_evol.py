import json
from utils.utils import read_data_file, get_req
import os
import argparse
from model import thread_parallel, get_model
from rouge_score import rouge_scorer
from functools import partial
from rouge_chinese import Rouge
import copy
from self_instruct import pass_rouge_list, str_is_cjk
import random


def createConstraintsPrompt(depth_instruction, instruction):
    prompt = depth_instruction.format("Please add one more constraints/requirements into #The Given Prompt#'")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt
def createDeepenPrompt(depth_instruction, instruction):
    prompt = depth_instruction.format("If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt
def createConcretizingPrompt(depth_instruction, instruction):
    prompt = depth_instruction.format("Please replace general concepts with more specific concepts.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt
def createReasoningPrompt(depth_instruction, instruction):
    prompt = depth_instruction.format("If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to more complicate which requires multiple-step reasoning.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt

def createOutputConstraintPrompt(depth_instruction, instruction):
    prompt = depth_instruction.format('Please add one more alternative output format constraints into #The Given Prompt#.')
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt

def createBreadthPrompt(instruction):
    prompt = "I want you act as a Prompt Creator.\r\n\
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
Remember: The rewritten prompt MUST use the same language as the #The Given Prompt# . Eg: If the #The Given Prompt# is written in Chinese, your '#Rewritten Prompt#' should also written in Chinese\r\n \
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
The language of the #Created Prompt# should be same as the #Given Prompt#.\e\n\
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"
    prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Created Prompt#:\r\n"
    return prompt

def get_evol_ins_normal():
    depth_instruction = "I want you act as a Prompt Rewriter.\r\n \
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
But the rewritten prompt must be reasonable and must be understood and responded by humans using pure text response.\r\n \
Remember: The rewritten prompt MUST use the same language as the #The Given Prompt#. Eg: If the #The Given Prompt# is written in Chinese, your '#Rewritten Prompt#' should also written in Chinese \r\n \
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input data provided by #The Given Prompt#. \r\n \
Remember: You should try your best to make the #Rewritten Prompt# become a complex problem, not a verbose problem. #Rewritten Prompt# can add more than 10 words or charactors into #The Given Prompt#. \r\n\
You should directly give the rewritten prompt starting with '#Rewritten Prompt#'. The phrase '#The Given Prompt#' and 'given prompt' are not allowed to appear in #Rewritten Prompt# \r\n\
You SHOULD complicate the given prompt using the following method: \r\n\
{}"
    
    evol_list = [createBreadthPrompt, partial(createConcretizingPrompt, depth_instruction), 
                 partial(createReasoningPrompt, depth_instruction), partial(createConstraintsPrompt, depth_instruction), 
                 partial(createDeepenPrompt, depth_instruction), partial(createOutputConstraintPrompt, depth_instruction)]
    return random.choice(evol_list)


def get_evol_ins_code():
    return lambda instruct: '''
Please increase the difficulty of the given programming question a bit.
But the rewritten prompt must be reasonable and must be understood and responded by humans using pure text response.
Remember:The rewritten prompt MUST also use the same language as the given prompt. Eg: If the #The Given Prompt# is written in Chinese, your '#Rewritten Prompt#' should also written in Chinese.
You can increase the difficulty using, but not limited to, the following methods and any of their combinations:
1. Add new constraints and requirements to the original problem, adding approximately 10 additional words.
2. Replace a commonly used requirement in the programming task with a less common and more specific one.
3. If the original problem can be solved with only a few logical steps, please add more reasoning steps.
4. Provide a piece of erroneous code as a reference to increase misdirection.
5. Propose higher time or space complexity requirements, but please refrain from doing so frequently.
Remember: You should directly give the rewritten prompt starting with '#Rewritten Prompt#'. The phrase '#The Given Prompt#' and 'given prompt' are not allowed to appear in #Rewritten Prompt#
#The Given Prompt#.
{}
#Rewritten Prompt#:'''.format(instruct)


def get_evol_ins_math():
    depth_instruction = "I want you act as a Prompt Rewriter.\r\n \
Your objective is to rewrite a given math problem prompt to let it more hard or complicate to handle.\r\n \
But the rewritten prompt must be reasonable and must be understood and responded by humans using pure text response.\r\n \
Remember: The rewritten prompt MUST also use the same language as the #The Given Prompt#. Eg: If the #The Given Prompt# is written in Chinese, your '#Rewritten Prompt#' should also written in Chinese\r\n \
If there are more condition or factor added when complicating #The Given Prompt#, you MUST provide ALL necessary required values for the condition to let the prompt could still be solved humans. \
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input data provided by #The Given Prompt#. \r\n \
You should directly give the rewritten prompt starting with '#Rewritten Prompt#'. The phrase '#The Given Prompt#' and 'given prompt' are not allowed to appear in #Rewritten Prompt# \r\n\
You SHOULD complicate the given prompt using the following method: \r\n\
{}"

    def createConstraintsPrompt(instruction):
        prompt = depth_instruction.format("Please switch the problem or function to solve to a more complicate one for the #The Given Prompt#'")
        prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
        prompt += "#Rewritten Prompt#:\r\n"
        return prompt
    
    evol_list = [createBreadthPrompt, partial(createReasoningPrompt, depth_instruction), 
                 partial(createConstraintsPrompt, depth_instruction), partial(createOutputConstraintPrompt, depth_instruction)]
    return random.choice(evol_list)


def check_prompts_pass(resp_list, original_prompt, scorer, model):
    # rouge_l filter -> answer filter
    cmp_list = [original_prompt]
    pass_answer = []

    answer_black_keyword = ['需要您提供', '请您提供', '请提供', '我需要了解', '请告诉我', '您没有提供', '由于我无', '任务 1', '任务1', '任务9', 'Task 9', 'Task 10', 'Task9', '作为一个AI', '作为一个大语言', 'As an AI', 'as an AI']
    answer_black_prefix = ['对不起', 'Sorry, ', '很抱歉', '非常抱歉']
    for resp in resp_list:
        resp_lan = '简中' if str_is_cjk(resp) else '英语'
        if pass_rouge_list(scorer=scorer, prompt_cjk_pair=(resp, resp_lan == '简中'), target_list=cmp_list, threshold=0.75):
            ans = None
            # try 3 times
            for _ in range(3):
                try:
                    ans = model.call(resp)
                except Exception as e:
                    print(f'Error when trying to get the response for evoled prompt. Error={e}')
                if ans:
                    break
            if ans is None:
                continue
            blacked = False
            for prefix in answer_black_prefix:
                if ans.startswith(prefix):
                    blacked = True
                    break
            if not blacked:
                for keyword in answer_black_keyword:
                    if keyword in ans:
                        blacked = True
                        break
            if not blacked:
                #print(f'evoled prompts={resp}, \t ans={ans}')
                cmp_list.append(resp)
                pass_answer.append((resp, ans))
    return pass_answer


def evol_instruct(data_evol_pair, model, scorer, current_epoch):
    prompt, evol_info = data_evol_pair
    failed_time, evol_epoch, data = evol_info
    if data['cls_res']['能力'] == '数学':
        evol_ins = get_evol_ins_math()
    elif data['cls_res']['能力'] == '代码':
        evol_ins = get_evol_ins_code()
    else:
        evol_ins = get_evol_ins_normal()
    try:
        # evol 3 times, if all failed, mark it as failed
        resp_list = []
        for _ in range(3):
            response = model.call(evol_ins(prompt))
            if '#Rewritten Prompt#:' in response:
                response = response.rsplit('#Rewritten Prompt#:', 1)[-1].strip()
            response = response.rsplit('#Rewritten Prompt#', 1)[-1]
            response = response.rsplit('#Created Prompt#', 1)[-1]
            resp_list.append(response)
        passed_check_prompts = check_prompts_pass(resp_list, prompt, scorer, model)
        if len(passed_check_prompts) == 0:
            return [(prompt, [failed_time + 1, evol_epoch, data])]
        else:
            out_datas = []
            for p, ans in passed_check_prompts:
                out_d = copy.deepcopy(data)
                out_d['prompt'] = p
                out_d['response'] = ans
                out_datas.append((p, [0, 0, out_d]))
            return [(prompt, [failed_time, current_epoch, data])] + out_datas
    except Exception as e:
        print(f'call model failed, error={e}')
        return [data_evol_pair]


def write_output_file(prompt_evol_infos, output_name):
    with open(output_name, 'w') as f_out:
        json.dump(prompt_evol_infos, f_out, indent=2, ensure_ascii=False)


def make_evol(input_data, output_name, model_name, thread_num, epoch):
    if len(input_data) > 0:
        get_req(input_data[0])
    req_prompts = {}
    for d in input_data:
        # failed_num, evol_epoch, d_original_info. If the data hasn't evoled, set the evol_epoch to 0
        req_prompts.setdefault(get_req(d), [0, 0, d])

    prompt_evol_infos = {}
    if os.path.exists(output_name):
        with open(output_name) as f_in:
            prompt_evol_infos = json.load(f_in)
    req_prompts.update(prompt_evol_infos)
    evol_input = []
    for k, v in req_prompts.items():
        if v[0] < 3 and v[1] < 1:
            evol_input.append((k, v))

    #(eng_scorer, cjk_scorer)
    scorer = (rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True), Rouge())
    model = get_model(model_name)
    for e in range(1, epoch + 1):
        evol_datas = thread_parallel(evol_instruct, evol_input, threads=thread_num, extra_paras=(model, scorer, e))
        num_new_prompts = 0
        for d in evol_datas:
            if d:
                new_num = len(d)
                num_new_prompts += new_num - 1
                for ele in d:
                    prompt_evol_infos[ele[0]] = ele[1]
                # save output per 100 new prompts
                if num_new_prompts > 100:
                    write_output_file(prompt_evol_infos, output_name)
                    num_new_prompts = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='wizard-evol to get more complex training pairs')
    parser.add_argument('-i', '--input', type=str, nargs='*', help='the input eval files, split multiple file in space')
    parser.add_argument('-o', '--output', type=str, default='output/wizard_evol', help='the output folder to store the final res')
    parser.add_argument('-m', '--model', type=str, default='gpt4-turbo', help="The seed gen model")
    parser.add_argument('-t', '--thread', type=int, default=40, help='the thread num when excuting the program')
    parser.add_argument('-e', '--epoch', type=int, default=2, help='The evol epochs what to loop')
    args=parser.parse_args()

    if isinstance(args.input, list):
        input_all_files = args.input
    else:
        input_all_files = [args.input]
    input_data = []
    for in_file in input_all_files:
        input_data.extend(read_data_file(in_file))
    make_evol(input_data, args.output, args.model, args.thread, args.epoch)