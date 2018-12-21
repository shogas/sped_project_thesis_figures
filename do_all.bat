@echo off

REM python vdf.py runs\vdf.txt
REM 
python combined_loading_map.py ..\..\Data\Runs\run_110_three_phase_no_split_20181211_12_13_40_528155 l2_norm_fourier 1 20

REM Both, to get loading map and orientations, template match first to get loading map from l2norm
python combined_loading_map.py ..\..\Data\Runs\run_110_full_nmf_20181211_21_18_30_538032 template_match
python combined_loading_map.py ..\..\Data\Runs\run_110_full_nmf_20181211_21_18_30_538032 l2_norm_fourier 2.6
python combined_loading_map.py ..\..\Data\Runs\run_110_full_umap_a_20181129_09_02_37_114197 template_match
python combined_loading_map.py ..\..\Data\Runs\run_110_full_umap_a_20181129_09_02_37_114197 l2_norm_fourier 1.526

python combined_loading_map.py ..\..\Data\Runs\run_110_full_nmf_cepstrum_20181216_16_11_25_759754 l2_norm 0.35

python combined_loading_map.py ..\..\Data\Runs\run_112_c_full_nmf_20181217_00_53_39_467258 l2_norm 2.38 500 90
python combined_loading_map.py ..\..\Data\Runs\run_112_d_full_nmf_20181217_11_48_44_542598 l2_norm 3 500 90
python combined_loading_map.py ..\..\Data\Runs\run_112_e_full_nmf_20181217_23_35_19_106171 l2_norm 3 500 90

python combined_loading_map.py ..\..\Data\Runs\run_112_c_full_nmf_cepstrum_20181218_17_20_30_787075 l2_norm 0.0 500 90
python combined_loading_map.py ..\..\Data\Runs\run_112_d_full_nmf_cepstrum_20181217_11_49_34_600266 l2_norm 0.2 500 90
python combined_loading_map.py ..\..\Data\Runs\run_112_e_full_nmf_cepstrum_20181217_23_35_19_106171 l2_norm 0.1 500 90

python combined_loading_map.py ..\..\Data\Runs\run_112_c_full_umap_20181217_00_53_39_467258 l2_norm 0 500 90
python combined_loading_map.py ..\..\Data\Runs\run_112_d_full_umap_20181217_11_49_02_191403 l2_norm 0 500 90
python combined_loading_map.py ..\..\Data\Runs\run_112_e_full_umap_20181217_23_35_27_335115 l2_norm 0 500 90

python combined_loading_map.py ..\..\Data\Runs\run_vdf_20181211_16_25_02_123927 l2_norm_fourier 0 20

REM 
python combined_orientation.py ..\..\Data\Runs\run_110_full_template_match_20181123_09_18_24_085171
python combined_performance.py correlate ..\..\data\Runs\run_perf_correlate
python combined_performance.py split_nmf ..\..\data\Runs\run_perf_split
python combined_performance.py split_umap ..\..\data\Runs\run_perf_split_umap
REM
REM python skree.py D:/Dokumenter/MTNANO/Prosjektoppgave/SPED_data_GaAs_NW/raw/Julie_180510_SCN45_FIB_a.blo ..\..\Data\Runs\run_skree full_110

REM python svg_crystal_generator.py zb ..\..\Data\Runs\run_svg_crystal\crystal_structure_zb.svg
REM python svg_crystal_generator.py zb_110 ..\..\Data\Runs\run_svg_crystal\crystal_structure_zb_110.svg
REM python svg_crystal_generator.py zb_112 ..\..\Data\Runs\run_svg_crystal\crystal_structure_zb_112.svg
REM python svg_crystal_generator.py wz ..\..\Data\Runs\run_svg_crystal\crystal_structure_wz.svg
REM python svg_crystal_generator.py wz_1120 ..\..\Data\Runs\run_svg_crystal\crystal_structure_wz_1120.svg
REM python svg_crystal_generator.py wz_1010 ..\..\Data\Runs\run_svg_crystal\crystal_structure_wz_1010.svg
REM python svg_crystal_generator.py sc ..\..\Data\Runs\run_svg_crystal\crystal_structure_sc.svg
REM python svg_crystal_generator.py fcc ..\..\Data\Runs\run_svg_crystal\crystal_structure_fcc.svg
REM python svg_crystal_generator.py bcc ..\..\Data\Runs\run_svg_crystal\crystal_structure_bcc.svg
REM python svg_crystal_generator.py hcp ..\..\Data\Runs\run_svg_crystal\crystal_structure_hcp.svg
REM python rotation_list_scatter.py cubic ..\..\Data\Runs\run_rotation_list_plot\cubic.tex
REM python rotation_list_scatter.py hexagonal ..\..\Data\Runs\run_rotation_list_plot\hexagonal.tex
REM 
REM FOR %%F IN (..\..\Data\Runs\run_svg_crystal\crystal_structure_*.svg) DO (
REM     ECHO Converting %%F to %%~pnF.pdf
REM     "C:\Program Files\Inkscape\inkscape.exe" %%F --export-pdf=%%~pnF.pdf --export-area-drawing
REM )
REM matlab -nosplash -nodisplay -nodesktop -r "run test_ebsd.m"

REM python diffraction_pattern_single.py runs\dp_single.txt

python figures.py runs\figures.txt



REM Old
REM python combined_loading_map.py ..\..\Data\Runs\run_112_c_full_nmf_20181203_15_03_43_409590 l2_norm_fourier 3.3 500 90
REM REM python combined_loading_map.py ..\..\Data\Runs\run_110_full_umap_a_20181129_09_02_37_114197 l2_norm 4.1
REM REM python combined_loading_map.py ..\..\Data\Runs\run_110_full_umap_a_20181126_23_17_20_750710 l2_norm_compare
REM REM python combined_loading_map.py ..\..\Data\Runs\run_112_c_full_nmf_20181202_14_45_40_881183 l2_norm_fourier 3.3 500 90
REM python combined_loading_map.py ..\..\Data\Runs\run_112_c_full_umap_20181203_22_18_40_476051 l2_norm 3 500 90
