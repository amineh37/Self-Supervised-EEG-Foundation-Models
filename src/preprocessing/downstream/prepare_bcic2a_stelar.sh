python prepare_bcic2a_stelar.py \
  --root_dir /projects/EEG-foundation-model/STELAR/data/downstream/BCIC2A_mat \
  --out_db /projects/EEG-foundation-model/STELAR/data/downstream/BCIC2a_processed/stelar \
  --band_low 0.3 --band_high 50 \
  --t_start 2 --window_sec 4 \
  --resample_hz 200
#   --apply_ems   # optional