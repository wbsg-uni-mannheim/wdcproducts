import os 

sizes = ['small', 'medium', 'large'] 
difficulties = ['20cc80rnd', '50cc50rnd', '80cc20rnd'] 
unseens = ['000un', '050un', '100un'] 

for seed in range(3):
    for size in sizes: 
        for difficulty in difficulties: 
            for unseen in unseens: 
                cmd = """CUDA_VISIBLE_DEVICES=3 python train.py \
                        --task final_%s_%s%s \
                        --run_id %d \
                        --batch_size 16 \
                        --max_len 256 \
                        --lr 5e-6 \
                        --n_epochs 50 \
                        --finetuning \
                        --split \
                        --lm roberta""" % (size, difficulty, unseen, seed)
                print(cmd)
                os.system(cmd)
               