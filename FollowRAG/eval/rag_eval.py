"""
response RAG score based on RAG document Q&A
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from copy import deepcopy
from tqdm import tqdm
from utils.call_llm import call_gpt
from utils.util import read_jsonl

# Fixed mini test set ID to avoid different randomness
ids_eval_mini=[6, 25, 27, 30, 32, 44, 46, 57, 67, 71, 73, 80, 81, 89, 94, 95, 99, 103, 104, 114, 127, 142, 159, 163, 166, 167, 175, 196, 203, 214, 216, 217, 220, 223, 224, 225, 228, 233, 234, 238, 250, 270, 273, 274, 276, 281, 284, 296, 300, 322, 323, 332, 344, 348, 352, 363, 367, 370, 373, 379, 387, 388, 389, 405, 410, 429, 432, 459, 464, 469, 470, 473, 511, 517, 546, 549, 558, 565, 570, 574, 580, 591, 603, 604, 616, 618, 623, 633, 643, 654, 660, 665, 666, 671, 683, 687, 688, 691, 692, 695, 700, 703, 711, 719, 721, 748, 759, 765, 770, 780, 787, 793, 808, 812, 814, 817, 823, 828, 831, 841, 842, 846, 856, 861, 863, 865, 869, 882, 903, 905, 916, 924, 930, 945, 946, 952, 953, 957, 969, 971, 973, 1000, 1005, 1014, 1019, 1031, 1046, 1048, 1070, 1071, 1082, 1090, 1094, 1108, 1132, 1133, 1138, 1145, 1148, 1162, 1164, 1179, 1186, 1197, 1200, 1205, 1212, 1219, 1221, 1229, 1240, 1241, 1243, 1245, 1249, 1251, 1252, 1262, 1266, 1267, 1274, 1280, 1297, 1298, 1310, 1313, 1323, 1340, 1342, 1344, 1352, 1356, 1361, 1375, 1376, 1379, 1380, 1389, 1393, 1398, 1402, 1407, 1415, 1432, 1451, 1455, 1458, 1459, 1460, 1462, 1469, 1470, 1472, 1477, 1482, 1495, 1496, 1499, 1500, 1511, 1535, 1543, 1558, 1561, 1568, 1570, 1587, 1590, 1594, 1596, 1618, 1619, 1622, 1625, 1634, 1635, 1640, 1642, 1643, 1648, 1654, 1655, 1671, 1685, 1692, 1703, 1721, 1738, 1747, 1762, 1788, 1799, 1810, 1812, 1816, 1820, 1832, 1833, 1841, 1853, 1859, 1865, 1873, 1878, 1884, 1888, 1892, 1897, 1898, 1914, 1920, 1926, 1943, 1949, 1952, 1953, 1955, 1963, 1969, 1984, 1990, 1993, 1998, 2002, 2013, 2044, 2046, 2057, 2061, 2063, 2064, 2069, 2074, 2075, 2078, 2084, 2085, 2091, 2093, 2095, 2103, 2108, 2109, 2136, 2140, 2142, 2145, 2152, 2170, 2174, 2175, 2177, 2183, 2194, 2202, 2206, 2209, 2214, 2218, 2222, 2230, 2233, 2234, 2235, 2237, 2251, 2252, 2259, 2261, 2265, 2308, 2309, 2315, 2318, 2344, 2350, 2352, 2357, 2367, 2368, 2370, 2371, 2378, 2383, 2388, 2391, 2406, 2407, 2409, 2415, 2421, 2423, 2441, 2451, 2457, 2466, 2473, 2478, 2505, 2513, 2529, 2533, 2537, 2540, 2548, 2552, 2568, 2569, 2600, 2617, 2618, 2635, 2641, 2650, 2656, 2658, 2664, 2665, 2666, 2667, 2674, 2676, 2678, 2683, 2692, 2696, 2697, 2708, 2719, 2726, 2734, 2736, 2745, 2758, 2760, 2772, 2773, 2780, 2785, 2793]

eval_prompt=\
"""
Please act as an impartial judge and perform the task: 
Given a [Question], you need to evaluate whether the [Response] correctly answers or hits the correct answer, and output your judgment after [Judge]. I will provide a correct answer [Reference] as a reference.

Scoring criteria:
- If the [Response] is completely correct and aligns with the correct answer, it scores 1 point; 
- If the [Response] partially answers correctly, it scores 0.5 point; 
- If the [response] is completely incorrect compared to the [Reference], it scores 0 point.

Note:
- Your only evaluation criterion is whether the [Response] correctly answered the answer, regardless of the format, language, case, length, etc., of the [Response]. Besides, providing more information than the [Reference] in the [Response] cannot be a reason for point deduction.
- Use the [Reference] as the correct answer reference rather than your own knowledge.
- The rating reply must strictly follow the format below: "Rating: [judge_score]\nReason: [judge_reason]", and do not output any other content. For example: "Rating: [0]\nReason: [Response and Reference are completely unrelated.]". Ensure that judge_score and judge_reason are enclosed in [].

[Question]
{question}

[Reference]
{answer_gold}

[Response]
{response}

[Judge]
"""

def construct_eval_prompt(dp,eval_column):
    input_prompt=eval_prompt.format(
        question=dp['question'],
        answer_gold=dp['answer_gold'],
        #response=dp[eval_column][0],
        response=dp[eval_column],
    )
    return input_prompt

def extract_rating(text):
    """Extract the Rating score from the evaluation results."""
    # Use regular expressions to match values in Rating (can only be 0, 0.5, or 1)
    match = re.search(r'Rating: \[(0|0\.5|1)\]', text)
    
    # Judge and return matching results
    if match:
        return float(match.group(1))
    else:
        return False

def gpt_rating(dp,eval_column):
    max_try=10 # If the extraction format is incorrect, try again
    cur_try=0
    input_prompt=construct_eval_prompt(dp,eval_column)
    while cur_try < max_try:
        try:
            output=call_gpt(llm_model_name='gpt-4o-0513',
                            message=input_prompt)
            if output==None:
                return (0,"judge error")
            rating_score=extract_rating(output)
            if rating_score is not False:
                return (rating_score,output)
            cur_try += 1
        except:
            cur_try+=1
    return (0,"judge error")

def rag_eval_main(eval_data,eval_type="all"):
    assert eval_type in ["mini","all"]
    eval_column='response'
    # run
    if eval_type=="mini":  # Only sample a portion for evaluation to save GPT call times
        eval_data=[eval_data[i] for i in ids_eval_mini]
    data_rag_evaled=[]
    for i in tqdm(range(len(eval_data))):
        dp=eval_data[i]
        gpt_score=gpt_rating(dp,eval_column)
        dp_rag_evaled=deepcopy(dp)
        dp_rag_evaled['rag_eval']={'gpt_rating_score':gpt_score[0],'gpt_rating_details':gpt_score[1]}
        data_rag_evaled.append(dp_rag_evaled)
    # count
    score_all=round(sum([dp['rag_eval']['gpt_rating_score'] for dp in data_rag_evaled])*100/len(data_rag_evaled),2)

    data_ifnq=[dp for dp in data_rag_evaled if "ifnq" in dp["type"]]
    score_ifnq=round(sum([dp['rag_eval']['gpt_rating_score'] for dp in data_ifnq])*100/len(data_ifnq),2)
    data_iftq=[dp for dp in data_rag_evaled if "iftq" in dp["type"]]
    score_iftq=round(sum([dp['rag_eval']['gpt_rating_score'] for dp in data_iftq])*100/len(data_iftq),2)
    data_ifhq=[dp for dp in data_rag_evaled if "ifhq" in dp["type"]]
    score_ifhq=round(sum([dp['rag_eval']['gpt_rating_score'] for dp in data_ifhq])*100/len(data_ifhq),2)
    data_ifwebq=[dp for dp in data_rag_evaled if "ifwebq" in dp["type"]]
    score_ifwebq=round(sum([dp['rag_eval']['gpt_rating_score'] for dp in data_ifwebq])*100/len(data_ifwebq),2)
    rag_eval_result={"score_all":score_all,"score_ifnq":score_ifnq,"score_iftq":score_iftq,"score_ifhq":score_ifhq,"score_ifwebq":score_ifwebq,"eval_num":len(data_rag_evaled)}
    print("rag_eval_result:",rag_eval_result)
    return data_rag_evaled,rag_eval_result

if __name__=='__main__':
    # test
    import os
    os.environ["OPENAI_API_KEY"]="your_openai_api_key"
    os.environ["OPENAI_API_BASE"]="your_openai_api_base"  # eg. https://api.openai.com/v1
    file_path="The path to the JSONL file that has already been inferred and needs to be evaluated, where the response field needs to be added to each field in `followRAG_full`."
    data=read_jsonl(file_path)
    rag_eval_main(data,eval_type="mini")


