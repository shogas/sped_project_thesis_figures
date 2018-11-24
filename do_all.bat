@echo off

python vdf.py runs\vdf.txt
python combined_loading_map.py ..\..\Data\Runs\run_110_full_nmf_20181123_21_51_25_665830 l2_norm 3.2
python figures.py runs\figures.txt
