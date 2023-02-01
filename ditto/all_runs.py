import os 

sizes = ['small', 'medium', 'large'] 
difficulties = ['20cc80rnd', '50cc50rnd', '80cc20rnd'] 
unseens = ['000un', '050un', '100un'] 

for seed in range(3):
    for size in sizes: 
        for difficulty in difficulties: 
            for unseen in unseens: 
                cmd = """CUDA_VISIBLE_DEVICES=2 python train_ditto.py \
                        --task final_%s_%s%s \
                        --logdir results_wdc3/ \
                        --run_id %d \
                        --batch_size 64 \
                        --max_len 256 \
                        --lr 5e-5 \
                        --n_epochs 50 \
                        --finetuning \
                        --lm roberta \
                        --da del""" % (size, difficulty, unseen, seed)
                print(cmd)
                os.system(cmd)