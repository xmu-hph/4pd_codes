nohup python run.py --png_save_path run4 --samples 500 >run4.log 2>&1 &

nohup python run.py --png_save_path run5 --samples 500 --epochs 50000 >run5.log 2>&1 &

nohup python run.py --png_save_path run6 --samples 500 --epochs 50000  --words 3 >run6.log 2>&1 &

nohup python run.py --png_save_path run7 --samples 500 --epochs 50000  --words 4 >run7.log 2>&1 &

nohup python run.py --png_save_path run8 --samples 500 --epochs 50000  --words 5 --cuda 'cuda:2' >run8.log 2>&1 &

nohup python run.py --png_save_path run9 --samples 500 --epochs 50000  --words 7 --cuda 'cuda:2' >run9.log 2>&1 &

nohup python run.py --png_save_path run10 --samples 500 --epochs 50000  --words 9 --cuda 'cuda:2' >run10.log 2>&1 &

nohup python run.py --png_save_path run11 --samples 500 --epochs 50000  --words 12 --cuda 'cuda:2' >run11.log 2>&1 &

nohup python run.py --png_save_path run12 --samples 1000 --epochs 50000  --words 2 --cuda 'cuda:2' >run12.log 2>&1 &

nohup python run.py --png_save_path run13 --samples 2000 --epochs 50000  --words 2 --cuda 'cuda:2' >run13.log 2>&1 &

nohup python run.py --png_save_path run14 --samples 4000 --epochs 50000  --words 2 --cuda 'cuda:2' >run14.log 2>&1 &