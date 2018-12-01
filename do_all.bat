@echo off

python vdf.py runs\vdf.txt
python combined_loading_map.py ..\..\Data\Runs\run_110_full_nmf_20181123_21_51_25_665830 l2_norm 3.2
python combined_loading_map.py ..\..\Data\Runs\run_110_full_umap_a_20181126_23_17_20_750710 l2_norm_compare
python combined_orientation.py ..\..\Data\Runs\run_110_full_template_match_20181123_09_18_24_085171
python combined_performance.py ..\..\data\Runs\run_perf_correlate
python combined_performance.py ..\..\data\Runs\run_perf_split
python svg_crystal_generator.py zb ../../data/Tmp/test.svg
python svg_crystal_generator.py wz ../../data/Tmp/test.svg
REM matlab -nosplash -nodisplay -nodesktop -r "run test_ebsd.m"
python figures.py runs\figures.txt
