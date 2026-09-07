import sys
import json
from crux.evaluation.rac_eval import rac_eval
from scipy import stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--run", type=str, required=True, help="Path to the run file")
    parser.add_argument("--qrel", type=str, required=True, help="Path to the qrel file")
    parser.add_argument("--judge", type=str, required=True, help="jsonl: {'id': str, 'docid': List[str], 'ratings': List[int]'}")
    parser.add_argument("--run_b", type=str, default=None)
    parser.add_argument("--filter_by_oracle", action="store_true", default=False)
    parser.add_argument("--run_a", type=str)
    parser.add_argument("--run_b", type=str)
    args = parser.parse_args()

    #### Start here
    from crux.tools import load_run_or_qrel, load_diversity_qrel, load_ratings
    qrel = load_run_or_qrel(args.qrel, threshold=1) # NOTE: only support binary labels so far
    div_qrel = load_diversity_qrel(args.qrel) # NOTE: only support binary labels so far
    ratings = load_ratings(args.judge)

    # load runs
    runa = load_run_or_qrel(args.run_a, topk=10)
    runb = load_run_or_qrel(args.run_b, topk=10)

    outputa = rac_eval(
        run=runa, 
        qrel=qrel, div_qrel=div_qrel, 
        run_b=None, # TODO: this feature hasnt been integrated yet.
        tau=3,
        cutoff=10,
        judge=ratings, 
        filter_by_oracle=args.filter_by_oracle, 
    )

    outputb = rac_eval(
        run=runb, 
        qrel=qrel, div_qrel=div_qrel, 
        run_b=None, # TODO: this feature hasnt been integrated yet.
        tau=3,
        cutoff=10,
        judge=ratings, 
        filter_by_oracle=args.filter_by_oracle, 
    )

#### End here
# print(f"### {args.input_run_1} > {args.input_run_2} | p-value   | ")
# for metric in ['nDCG', 'P', 'alpha_nDCG', 'coverage']:
#     list1 = output1[metric]
#     list2 = output2[metric]
#     t_stat, p_value_two_tailed = stats.ttest_rel(list1, list2)
#     if t_stat > 0:
#         if p_value_two_tailed/2 < 0.1:
#             print(f"### {metric}: one-tailed p-value = {p_value_two_tailed/2}")
#     else:
#         if (1-p_value_two_tailed/2) < 0.1:
#             print(f"### {metric}:  one-tailed p-value = {1-p_value_two_tailed/2}")
