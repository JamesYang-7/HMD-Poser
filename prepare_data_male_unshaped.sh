# # protocol 1
# python ./prepare_data_male_unshaped.py \
#     --protocol protocol1 \
#     --support_dir ./body_models/ \
#     --root_dir ./data/AMASS/ \
#     --save_dir ./datasets_processed/amass_p1/

# protocol 2
python prepare_data_male_unshaped.py \
  --protocol protocol2 \
  --support_dir ./body_models \
  --root_dir ./data/AMASS \
  --save_dir ./datasets_processed/amass_p2