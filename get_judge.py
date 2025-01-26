from judge import *
from prompts import prompt_judge_student, logic_are_same, prompt_rm_ranker_v4, logic_math, prompt_rm_ranker_BC, prompt_eval_math_BC, prompt_eval_no_math
from prompts import prompt_rm_model_ranker, prompt_rm_ranker_truth, prompt_distribution_model

def init_prompts():
    ALL_JUDGES = {}
    ALL_JUDGES['math_eval'] = OneStageJudge('gpt4-turbo-judgestudent', 
                                            prompt=prompt_judge_student.PROMPT_JUDGE, 
                                            parse_func=prompt_judge_student.parse)
    ALL_JUDGES['judge_same'] = RepeatJudge('gpt4-turbo-judgesame', 
                                            prompt=logic_are_same.PROMPT_JUDGE, 
                                            parse_func=logic_are_same.parse, 
                                            repeat_num=3)
    ALL_JUDGES['rm_ranker'] = CompareJudge('gpt4-turbo-judgesame', 
                                           prompt=prompt_rm_ranker_v4.PROMPT, 
                                           parse_func=prompt_rm_ranker_v4.parse, 
                                           reverse_func=prompt_rm_ranker_v4.reverse_func, 
                                           combine_func=prompt_rm_ranker_v4.combine_func, 
                                           timeout=240)
    ALL_JUDGES['BC_model_judge_correct'] = RepeatJudge('bc_online_eval', 
                                            prompt=prompt_eval_math_BC.PROMPT_JUDGE, 
                                            parse_func=prompt_eval_math_BC.parse, 
                                            repeat_num=3)
    ALL_JUDGES['gpt4_judge_no_answer'] = OneStageJudge('gpt4-turbo-judgesame', 
                                            prompt=prompt_eval_no_math.PROMPT, 
                                            parse_func=prompt_eval_no_math.parse)
    ALL_JUDGES['logic_math_two_stage'] = TwoStageJudge('gpt4-turbo-autojudge', 
                                                        expert_num=3, 
                                                        prompt=logic_math.PROMPT_JUDGE, 
                                                        prompt_stage2=logic_math.PROMPT_CHECK, 
                                                        parse_func=logic_math.parse)

    ALL_JUDGES['rm_ranker_with_cost'] = CompareJudge('gpt4-turbo-autojudge', 
                                                    prompt=prompt_rm_ranker_v4.PROMPT, 
                                                    parse_func=prompt_rm_ranker_v4.parse, 
                                                    reverse_func=prompt_rm_ranker_v4.reverse_func, 
                                                    combine_func=prompt_rm_ranker_v4.combine_func, 
                                                    return_details=True,
                                                    retry_num=5,
                                                    timeout=240)
    
    ALL_JUDGES['bc_rm_model_judge'] = OneStageJudge('bc_ranker', 
                                                    prompt=prompt_rm_model_ranker.PROMPT_JUDGE, 
                                                    parse_func=prompt_rm_model_ranker.parse,
                                                    retry_num=5)
    ALL_JUDGES['bc_rm_model_truth'] = OneStageJudge('bc_online_current', 
                                                    prompt=prompt_rm_ranker_truth.PROMPT_JUDGE, 
                                                    parse_func=prompt_rm_ranker_truth.parse,
                                                    retry_num=5,
                                                    timeout=240)
    ALL_JUDGES['distribution_model'] = OneStageJudge('distribution_model', 
                                                    prompt=prompt_distribution_model.PROMPT_JUDGE, 
                                                    parse_func=prompt_distribution_model.parse,
                                                    retry_num=5)

    ALL_JUDGES['rm_ranker_with_cost_new'] = CompareJudge('gpt4-turbo-autojudge', 
                                                        prompt=prompt_rm_ranker_BC.PROMPT, 
                                                        parse_func=prompt_rm_ranker_BC.parse, 
                                                        reverse_func=prompt_rm_ranker_BC.reverse_func, 
                                                        combine_func=prompt_rm_ranker_BC.combine_func,
                                                        return_details=True,
                                                        retry_num=5,
                                                        timeout=240)

    ALL_JUDGES['bc_rm_ranker_with_cost'] = CompareJudge('bc_test_1215', 
                                                    prompt=prompt_rm_ranker_BC.PROMPT, 
                                                    parse_func=prompt_rm_ranker_BC.parse, 
                                                    reverse_func=prompt_rm_ranker_BC.reverse_func, 
                                                    combine_func=prompt_rm_ranker_BC.combine_func,
                                                    return_details=True,
                                                    retry_num=5,
                                                    timeout=240)
    return ALL_JUDGES

def get_judge(judge_name):
    global ALL_JUDGES
    if 'ALL_JUDGES' not in globals():
        ALL_JUDGES = init_prompts()
    if judge_name in ALL_JUDGES.keys():
        return ALL_JUDGES[judge_name]
    else:
        print(f'judge {judge_name} not registered')
    return None
