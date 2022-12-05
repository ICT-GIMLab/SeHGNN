# (1,)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ACM --metapaths a p f ap pa pf fp --num-heads 8 --seeds 1 2 3 4 5 > ACM_1__01.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --dataset ACM --metapaths ap pa pf fp --num-heads 8 --seeds 1 2 3 4 5 > ACM_1__1.txt &!

# (1, 1)
CUDA_VISIBLE_DEVICES=2 python main.py --dataset ACM --metapaths a p f ap pa pf fp --num-heads 8 8 --seeds 1 2 3 4 5 > ACM_1_1__01.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --dataset ACM --metapaths ap pa pf fp --num-heads 8 8 --seeds 1 2 3 4 5 > ACM_1_1__1.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --dataset ACM --metapaths p ap pa pf fp --num-heads 8 8 --seeds 1 2 3 4 5 > ACM_1_1__t1.txt &!

# (1, 1, 1)
CUDA_VISIBLE_DEVICES=2 python main.py --dataset ACM --metapaths a p f ap pa pf fp --num-heads 8 8 8 --seeds 1 2 3 4 5 > ACM_1_1_1__01.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --dataset ACM --metapaths ap pa pf fp --num-heads 8 8 8 --seeds 1 2 3 4 5 > ACM_1_1_1__1.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ACM --metapaths p ap pa pf fp --num-heads 8 8 8 --seeds 1 2 3 4 5 > ACM_1_1_1__t1.txt &!

# (1, 1, 1, 1)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ACM --metapaths a p f ap pa pf fp --num-heads 8 8 8 8 --seeds 1 2 3 4 5 > ACM_1_1_1_1__01.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --dataset ACM --metapath ap pa pf fp --num-heads 8 8 8 8 --seeds 1 2 3 4 5 > ACM_1_1_1_1__1.txt &!
CUDA_VISIBLE_DEVICES=2 python main.py --dataset ACM --metapath p ap pa pf fp --num-heads 8 8 8 8 --seeds 1 2 3 4 5 > ACM_1_1_1_1__t1.txt &!

# (2)
CUDA_VISIBLE_DEVICES=3 python main.py --dataset ACM --metapaths p pa pf pap pfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_2__012.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --dataset ACM --metapaths p pap pfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_2__02.txt &!
CUDA_VISIBLE_DEVICES=2 python main.py --dataset ACM --metapaths pa pf pap pfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_2__12.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --dataset ACM --metapaths pap pfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_2__2.txt &!

# (2, 2)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ACM --metapaths p pa ap pf fp pap pfp apa apf fpf fpa --num-heads 8 8 --seeds 1 2 3 4 5 > ACM_2_2__012.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ACM --metapaths p pap pfp --num-heads 8 8 --seeds 1 2 3 4 5 > ACM_2_2__02.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --dataset ACM --metapaths pa ap pf fp pap pfp apa apf fpf fpa --num-heads 8 8 --seeds 1 2 3 4 5 > ACM_2_2__12.txt &!
CUDA_VISIBLE_DEVICES=2 python main.py --dataset ACM --metapaths pap pfp --num-heads 8 8 --seeds 1 2 3 4 5 > ACM_2_2__2.txt &!

# (3)
CUDA_VISIBLE_DEVICES=3 python main.py --dataset ACM --metapaths p pa pf pap pfp papa papf pfpa pfpf --num-heads 8 --seeds 1 2 3 4 5 > ACM_3__0123.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --dataset ACM --metapaths pa pf pap pfp papa papf pfpa pfpf --num-heads 8 --seeds 1 2 3 4 5 > ACM_3__123.txt &!
CUDA_VISIBLE_DEVICES=2 python main.py --dataset ACM --metapaths pap pfp papa papf pfpa pfpf --num-heads 8 --seeds 1 2 3 4 5 > ACM_3__23.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --dataset ACM --metapaths papa papf pfpa pfpf --num-heads 8 --seeds 1 2 3 4 5 > ACM_3__3.txt &!

# (4)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ACM --metapaths p pa pf pap pfp papa papf pfpa pfpf papap papfp pfpap pfpfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_4__01234.txt &!
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ACM --metapaths p pap pfp papa papf pfpa pfpf papap papfp pfpap pfpfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_4__0234.txt &!
CUDA_VISIBLE_DEVICES=1 python main.py --dataset ACM --metapaths p pap pfp papap papfp pfpap pfpfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_4__024.txt &!
CUDA_VISIBLE_DEVICES=2 python main.py --dataset ACM --metapaths pa pf pap pfp papa papf pfpa pfpf papap papfp pfpap pfpfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_4__1234.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --dataset ACM --metapaths pap pfp papa papf pfpa pfpf papap papfp pfpap pfpfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_4__234.txt &!
CUDA_VISIBLE_DEVICES=3 python main.py --dataset ACM --metapaths pap pfp papap papfp pfpap pfpfp --num-heads 8 --seeds 1 2 3 4 5 > ACM_4__24.txt &!