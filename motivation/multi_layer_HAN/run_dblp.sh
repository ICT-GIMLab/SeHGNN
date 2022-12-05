# (1)
CUDA_VISIBLE_DEVICES=0 python main.py --metapaths a p c ap pa tp pt cp pc --num-heads 8 --seeds 1 2 3 4 5 > 1__01.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --metapaths ap pa tp pt cp pc --num-heads 8 --seeds 1 2 3 4 5 > 1__1.txt &!

# (1, 1)
CUDA_VISIBLE_DEVICES=2 python main.py --metapaths a p c ap pa tp pt cp pc --num-heads 8 8 --seeds 1 2 3 4 5 > 1_1__01.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --metapaths ap pa tp pt cp pc --num-heads 8 8 --seeds 1 2 3 4 5 > 1_1__1.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --metapaths a ap pa tp pt cp pc --num-heads 8 8 --seeds 1 2 3 4 5 > 1_1__t1.txt &!

# (1, 1, 1)
CUDA_VISIBLE_DEVICES=2 python main.py --metapaths a p c ap pa tp pt cp pc --num-heads 8 8 8 --seeds 1 2 3 4 5 > 1_1_1__01.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --metapaths ap pa tp pt cp pc --num-heads 8 8 8 --seeds 1 2 3 4 5 > 1_1_1__1.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --metapaths a ap pa tp pt cp pc --num-heads 8 8 8 --seeds 1 2 3 4 5 > 1_1_1__t1.txt &!

# (1, 1, 1, 1)
CUDA_VISIBLE_DEVICES=0 python main.py --metapaths a p c ap pa tp pt cp pc --num-heads 8 8 8 8 --seeds 1 2 3 4 5 > 1_1_1_1__01.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --metapath ap pa tp pt cp pc --num-heads 8 8 8 8 --seeds 1 2 3 4 5 > 1_1_1_1__1.txt &!
CUDA_VISIBLE_DEVICES=2 python main.py --metapath a ap pa tp pt cp pc --num-heads 8 8 8 8 --seeds 1 2 3 4 5 > 1_1_1_1__t1.txt &!

# (2)
CUDA_VISIBLE_DEVICES=3 python main.py --metapaths a ap apa apc apt --num-heads 8 --seeds 1 2 3 4 5 > 2__012.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --metapaths ap apa apc apt --num-heads 8 --seeds 1 2 3 4 5 > 2__12.txt &!
CUDA_VISIBLE_DEVICES=2 python main.py --metapaths apa apc apt --num-heads 8 --seeds 1 2 3 4 5 > 2__2.txt &!

# (2, 2)
CUDA_VISIBLE_DEVICES=1 python main.py --metapaths a p c ap pa tp pt cp pc apa apc cpa apt tpa --num-heads 8 8 --seeds 1 2 3 4 5 > 2_2__012.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --metapaths a p c apa apc cpa apt tpa --num-heads 8 8 --seeds 1 2 3 4 5 > 2_2__02.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --metapaths ap pa tp pt cp pc apa apc cpa apt tpa --num-heads 8 8 --seeds 1 2 3 4 5 > 2_2__12.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --metapaths apa apc cpa apt tpa --num-heads 8 8 --seeds 1 2 3 4 5 > 2_2__2.txt &!

CUDA_VISIBLE_DEVICES=1 python main.py --metapaths a p c ap pa tp pt cp pc apa apc cpa apt tpa pap pcp ptp --num-heads 8 8 --seeds 1 2 3 4 5 > 2_2__012n.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --metapaths a p c apa apc cpa apt tpa pap pcp ptp --num-heads 8 8 --seeds 1 2 3 4 5 > 2_2__02n.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --metapaths ap pa tp pt cp pc apa apc cpa apt tpa pap pcp ptp --num-heads 8 8 --seeds 1 2 3 4 5 > 2_2__12n.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --metapaths apa apc cpa apt tpa pap pcp ptp --num-heads 8 8 --seeds 1 2 3 4 5 > 2_2__2n.txt &!

# (3)
CUDA_VISIBLE_DEVICES=2 python main.py --metapaths a ap apa apc apt aptp apcp --num-heads 8 --seeds 1 2 3 4 5 > 3__0123.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --metapaths ap apa apc apt aptp apcp --num-heads 8 --seeds 1 2 3 4 5 > 3__123.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --metapaths apa apc apt aptp apcp --num-heads 8 --seeds 1 2 3 4 5 > 3__23.txt &!
CUDA_VISIBLE_DEVICES=2 python main.py --metapaths aptp apcp --num-heads 8 --seeds 1 2 3 4 5 > 3__3.txt &!

# (4)
CUDA_VISIBLE_DEVICES=1 python main.py --metapaths a ap apa apc apt aptp apcp aptpa apcpa apapa --num-heads 8 --seeds 1 2 3 4 5 > 4__01234.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --metapaths a apa apc apt aptp apcp aptpa apcpa apapa --num-heads 8 --seeds 1 2 3 4 5 > 4__0234.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --metapaths a apa apc apt aptpa apcpa apapa --num-heads 8 --seeds 1 2 3 4 5 > 4__024.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --metapaths ap apa apc apt aptp apcp aptpa apcpa apapa --num-heads 8 --seeds 1 2 3 4 5 > 4__1234.txt &!
CUDA_VISIBLE_DEVICES=2 python main.py --metapaths apa apc apt aptp apcp aptpa apcpa apapa --num-heads 8 --seeds 1 2 3 4 5 > 4__234.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --metapaths apa apc apt aptpa apcpa apapa --num-heads 8 --seeds 1 2 3 4 5 > 4__24.txt &!