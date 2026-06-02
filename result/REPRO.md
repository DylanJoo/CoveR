# BEIR

| Model | Dataset | Metric | Score |
|---|---|---|---|
| relevance-10k | arg | nDCG@10 | 0.3407 |
| relevance-10k | sci | nDCG@10 | 0.7112 |
| relevance-10k | tre | nDCG@10 | 0.8223 |
| relevance-10k | web | nDCG@10 | 0.2582 |
| relevance-10k | cli | nDCG@10 | 0.2725 |
| relevance-10k | dbp | nDCG@10 | 0.3866 |
| relevance-10k | fev | nDCG@10 | 0.7711 |
| relevance-10k | fiq | nDCG@10 | 0.3994 |
| relevance-10k | hot | nDCG@10 | 0.6310 |
| relevance-10k | nfc | nDCG@10 | 0.3338 |
| relevance-10k | nq  | nDCG@10 | 0.5494 |
| relevance-10k | quo | nDCG@10 | 0.8528 |
| relevance-10k | sci | nDCG@10 | 0.1810 |
| relevance-10k | avg | nDCG@10 | 0.5008 |
|---|---|---|---|
| cover-5k | arg | nDCG@10	| 0.3669 |  
| cover-5k | sci | nDCG@10	| 0.7005 |
| cover-5k | tre | nDCG@10	| 0.8105 |
| cover-5k | web | nDCG@10	| 0.2584 |
| cover-5k | cli | nDCG@10	| 0.2616 |
| cover-5k | dbp | nDCG@10	| 0.3914 |
| cover-5k | fev | nDCG@10	| 0.7813 |
| cover-5k | fiq | nDCG@10	| 0.3967 |
| cover-5k | hot | nDCG@10	| 0.6199 |
| cover-5k | nfc | nDCG@10	| 0.3451 |
| cover-5k | nq  | nDCG@10	| 0.5596 |
| cover-5k | quo | nDCG@10	| 0.8401 |
| cover-5k | sci | nDCG@10	| 0.1987 |
| cover-5k | avg | nDCG@10  | 0.5024 |
|---|---|---|---|
| cover-5k | arg | nDCG@10 | 0.3673 |
| cover-5k | sci | nDCG@10 | 0.7051 |
| cover-5k | tre | nDCG@10 | 0.7739 |
| cover-5k | web | nDCG@10 | 0.2571 |
| cover-5k | cli | nDCG@10 | 0.2603 |
| cover-5k | dbp | nDCG@10 | 0.3909 |
| cover-5k | fev | nDCG@10 | 0.7779 |
| cover-5k | fiq | nDCG@10 | 0.4009 |
| cover-5k | hot | nDCG@10 | 0.6217 |
| cover-5k | nfc | nDCG@10 | 0.3468 |
| cover-5k | nq  | nDCG@10 | 0.5585 |
| cover-5k | quo | nDCG@10 | 0.8507 |
| cover-5k | sci | nDCG@10 | 0.1986 |
| cover-5k | avg | nDCG@10 | 0.5007 |
|---|---|---|---|
| scope-flt-cover-5k | arg | nDCG@10 | 0.3508 |
| scope-flt-cover-5k | sci | nDCG@10 | 0.7032 |
| scope-flt-cover-5k | tre | nDCG@10 | 0.6566 |
| scope-flt-cover-5k | web | nDCG@10 | 0.2001 |
| scope-flt-cover-5k | cli | nDCG@10 | 0.2611 |
| scope-flt-cover-5k | dbp | nDCG@10 | 0.3693 |
| scope-flt-cover-5k | fev | nDCG@10 | 0.7980 |
| scope-flt-cover-5k | fiq | nDCG@10 | 0.3868 |
| scope-flt-cover-5k | hot | nDCG@10 | 0.6144 |
| scope-flt-cover-5k | nfc | nDCG@10 | 0.3325 |
| scope-flt-cover-5k | nq  | nDCG@10 | 0.5035 |
| scope-flt-cover-5k | quo | nDCG@10 | 0.8571 |
| scope-flt-cover-5k | sci | nDCG@10 | 0.1988 |
| scope-flt-cover-5k | avg | nDCG@10 | 0.4794 |
|---|---|---|---|

# NeuCLIR and CRUX
| Model | Dataset | Metric | Score |
|---|---|---|---|
| cover-5k            | neuclir24-test.trec | P@10 | 0.8421 | nDCG@10 | 0.8611 | alpha_nDCG@10 | 0.5755 | Cov@10 | 0.6722 | 
| relevance-10k       | neuclir24-test.trec | P@10 | 0.6947 | nDCG@10 | 0.7230 | alpha_nDCG@10 | 0.4594 | Cov@10 | 0.5540 | 
| scope-5k            | neuclir24-test.trec | P@10 | 0.8263 | nDCG@10 | 0.8369 | alpha_nDCG@10 | 0.5888 | Cov@10 | 0.6769 | 
| scope-flt-cover-5k  | neuclir24-test.trec | P@10 | 0.8263 | nDCG@10 | 0.8445 | alpha_nDCG@10 | 0.6023 | Cov@10 | 0.6729 | 
| scope-flt-cover-10k | neuclir24-test.trec | P@10 | 0.8316 | nDCG@10 | 0.8379 | alpha_nDCG@10 | 0.6044 | Cov@10 | 0.6963 | 

| Model | Dataset | Metric | Score |
|---|---|---|---|
| cover-5k            | crux-mds-duc04      | P@10 | 0.6860 | nDCG@10 | 0.7133 | alpha_nDCG@10 | 0.5792 | Cov@10 | 0.6308 | 
| cover-5k            | crux-mds-multi_news | P@10 | 0.3750 | nDCG@10 | 0.5514 | alpha_nDCG@10 | 0.5774 | Cov@10 | 0.5844 | 
| relevance-10k       | crux-mds-duc04      | P@10 | 0.6280 | nDCG@10 | 0.6481 | alpha_nDCG@10 | 0.5188 | Cov@10 | 0.5843 | 
| relevance-10k       | crux-mds-multi_news | P@10 | 0.3390 | nDCG@10 | 0.4872 | alpha_nDCG@10 | 0.4948 | Cov@10 | 0.5176 | 
