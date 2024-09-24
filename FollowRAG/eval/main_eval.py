import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from eval.if_eval import if_eval_main
from eval.rag_eval import rag_eval_main
from utils.util import read_jsonl,save_jsonl,get_current_time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, 
                        help="The path to the JSONL file that has already been inferred and needs to be evaluated, where the response field needs to be added to each field in `followRAG_full`.")
    parser.add_argument("--rag_eval_type", type=str, choices=["mini","all"],
                        help="Since the evaluation of RAG scores requires calling GPT, this parameter is used to specify the type of RAG evaluation. 'mini' indicates that only a sample of the evaluation will be conducted.")
    parser.add_argument("--output_file_path", type=str, help="The path to save the evaluation results")
    parser.add_argument("--result_log_file_path", type=str, help="The path to save the result score log")
    args = parser.parse_args()
    return args

def main(args):
    log={}
    log["time"]=get_current_time()
    log["eval_file"]=args.input_file_path
    print("eval:",args.input_file_path)
    # eval
    data=read_jsonl(read_file_path=args.input_file_path)
    data_if_evaled,if_eval_result=if_eval_main(data)
    data_if_rag_evaled,rag_eval_result=rag_eval_main(data_if_evaled,eval_type=args.rag_eval_type)
    # marge data_if_evaled and data_if_rag_evaled
    data_evaled=[]
    data_if_rag_evaled= {item['key']: item for item in data_if_rag_evaled}
    for item in data_if_evaled:
        if item['key'] in data_if_rag_evaled:
            data_evaled.append(data_if_rag_evaled[item['key']])
        else:
            data_evaled.append(item)
    # log
    log["eval_score"]={"all":{"if":if_eval_result["all"]["loose_instruction_score"],
                              "rag":rag_eval_result["score_all"],
                              "avg":round((if_eval_result["all"]["loose_instruction_score"]+rag_eval_result["score_all"])/2,2)},
                       "ifnq":{"if":if_eval_result["ifnq"]["loose_instruction_score"],
                               "rag":rag_eval_result["score_ifnq"],
                               "avg":round((if_eval_result["all"]["loose_instruction_score"]+rag_eval_result["score_ifnq"])/2,2)},
                       "iftq":{"if":if_eval_result["iftq"]["loose_instruction_score"],
                               "rag":rag_eval_result["score_iftq"],
                               "avg":round((if_eval_result["iftq"]["loose_instruction_score"]+rag_eval_result["score_iftq"])/2,2)},
                       "ifhq":{"if":if_eval_result["ifhq"]["loose_instruction_score"],
                               "rag":rag_eval_result["score_ifhq"],
                               "avg":round((if_eval_result["ifhq"]["loose_instruction_score"]+rag_eval_result["score_ifhq"])/2,2)},
                       "ifwebq":{"if":if_eval_result["ifwebq"]["loose_instruction_score"],
                                 "rag":rag_eval_result["score_ifwebq"],
                                 "avg":round((if_eval_result["ifwebq"]["loose_instruction_score"]+rag_eval_result["score_ifwebq"])/2,2)},
                       }
    log["if_eval_result"]=if_eval_result
    log["rag_eval_result"]=rag_eval_result
    log["detailed_args"]=args.__dict__
    print(log)
    # save eval results
    save_jsonl(args.output_file_path,data_evaled,mode='w')
    save_jsonl(args.result_log_file_path,[log],mode='a')

if __name__=="__main__":
    args=get_args()
    main(args)