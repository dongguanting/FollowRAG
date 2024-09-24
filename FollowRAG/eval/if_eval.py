"""
Evaluate the instruction following score of the response
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.util import read_jsonl
from copy import deepcopy
from utils.instruction_following_eval import instructions_registry

def eval_instruction_following_strict(dp):
    """Strictly evaluate the instruction following of a data point."""
    response=dp['response']
    instruction_list=dp['instruction_id_list']
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**dp['kwargs'][index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=dp['prompt'])

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return is_following_list

def eval_instruction_following_loose(dp):
  """Loosely evaluate the instruction following of a data point."""
  response = dp['response']
  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
  revised_response_remove_first = response_remove_first.replace("*", "")
  revised_response_remove_last = response_remove_last.replace("*", "")
  revised_response_remove_both = response_remove_both.replace("*", "")
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  instruction_list = dp['instruction_id_list']
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**dp['kwargs'][index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=dp['prompt'])

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return is_following_list

def if_eval_main(data):
    data_if_evaled=[]
    for dp in data:
        # Evaluate each sample.
        strict_is_following_list=eval_instruction_following_strict(dp)
        loose_is_following_list=eval_instruction_following_loose(dp)
        dp_if_evaled=deepcopy(dp)
        dp_if_evaled['if_eval']={'strict_follow_all_instructions':all(strict_is_following_list),
                        'loose_follow_all_instructions':all(loose_is_following_list),
                        'strict_follow_instructions':strict_is_following_list,
                        'loose_follow_instructions':loose_is_following_list
                        }
        data_if_evaled.append(dp_if_evaled)
    # Calculate the total score.
    def calc_score(data_if_evaled):
        strict_prompt_score=round(sum([dp['if_eval']['strict_follow_all_instructions'] for dp in data_if_evaled])*100/len(data_if_evaled),2)
        loose_prompt_score=round(sum([dp['if_eval']['loose_follow_all_instructions'] for dp in data_if_evaled])*100/len(data_if_evaled),2)
        strict_follow_instructions,loose_follow_instructions=[],[]
        for dp in data_if_evaled:
            strict_follow_instructions+=dp['if_eval']['strict_follow_instructions']
            loose_follow_instructions+=dp['if_eval']['loose_follow_instructions']
        strict_instruction_score=round(sum(strict_follow_instructions)*100/len(strict_follow_instructions),2)
        loose_instruction_score=round(sum(loose_follow_instructions)*100/len(loose_follow_instructions),2)
        if_eval_score={'strict_prompt_score':strict_prompt_score,'loose_prompt_score':loose_prompt_score,'strict_instruction_score':strict_instruction_score,'loose_instruction_score':loose_instruction_score}
        return if_eval_score
    # Calculate the score for each category of IFNQ.
    if_eval_scores={"all":calc_score(data_if_evaled),
                    "ifnq":calc_score([dp for dp in data_if_evaled if 'ifnq' in dp['type']]),
                    "iftq":calc_score([dp for dp in data_if_evaled if 'iftq' in dp['type']]),
                    "ifhq":calc_score([dp for dp in data_if_evaled if 'ifhq' in dp['type']]),
                    "ifwebq":calc_score([dp for dp in data_if_evaled if 'ifwebq' in dp['type']])
                    }
    print("if_eval_scores:",if_eval_scores)
    return data_if_evaled,if_eval_scores

if __name__=="__main__":
    file_path="The path to the JSONL file that has already been inferred and needs to be evaluated, where the response field needs to be added to each field in `followRAG_full`."
    data=read_jsonl(file_path)
    if_eval_main(data)
