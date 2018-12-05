@echo off

REM python vdf.py runs\vdf_full_110.txt
REM python vdf.py runs\vdf_full_112_c.txt

python combined_loading_map.py ..\..\Data\Runs\run_110_three_phase_no_split_20181203_16_51_45_034518 l2_norm_fourier 0 20
python combined_loading_map.py ..\..\Data\Runs\run_110_full_nmf_20181123_21_51_25_665830 l2_norm_fourier 3.2
python combined_loading_map.py ..\..\Data\Runs\run_110_full_umap_a_20181126_23_17_20_750710 l2_norm_compare
REM python combined_loading_map.py ..\..\Data\Runs\run_112_c_full_nmf_20181202_14_45_40_881183 l2_norm_fourier 3.3 100 90
python combined_loading_map.py ..\..\Data\Runs\run_112_c_full_nmf_20181203_15_03_43_409590 l2_norm_fourier 3.3 100 90
python combined_loading_map.py ..\..\Data\Runs\run_112_c_full_umap_20181203_22_18_40_476051 l2_norm 3 100 90

python combined_orientation.py ..\..\Data\Runs\run_110_full_template_match_20181123_09_18_24_085171
python combined_performance.py correlate ..\..\data\Runs\run_perf_correlate
python combined_performance.py split_nmf ..\..\data\Runs\run_perf_split

python svg_crystal_generator.py zb ..\..\Data\Runs\run_svg_crystal\crystal_structure_zb.pdf
python svg_crystal_generator.py wz ..\..\Data\Runs\run_svg_crystal\crystal_structure_wz.pdf
python svg_crystal_generator.py sc ..\..\Data\Runs\run_svg_crystal\crystal_structure_sc.pdf
python svg_crystal_generator.py fcc ..\..\Data\Runs\run_svg_crystal\crystal_structure_fcc.pdf
python svg_crystal_generator.py bcc ..\..\Data\Runs\run_svg_crystal\crystal_structure_bcc.pdf
python svg_crystal_generator.py hcp ..\..\Data\Runs\run_svg_crystal\crystal_structure_hcp.pdf

FOR %%F IN (..\..\Data\Runs\run_svg_crystal\crystal_structure_*.svg) DO (
    inkscape %%F --export-pdf=%%~pnF.pdf --export-area-drawing
)
REM matlab -nosplash -nodisplay -nodesktop -r "run test_ebsd.m"
python figures.py runs\figures.txt
