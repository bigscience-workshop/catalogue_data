DEBUG_LOG=/home/lucile/logs.txt
echo -n $(date) >> $DEBUG_LOG

conda activate datacatalog

CATALOGUE_DATA_REPO="/home/lucile/code/catalogue_data"

cd $CATALOGUE_DATA_REPO

WIKI_DS=('lm_ar_wikipedia' 'lm_ar_wikisource_filtered' 'lm_ar_wikinews_filtered' 'lm_ar_wikibooks_filtered' 'lm_ar_wikiquote_filtered' 'lm_ar_wikiversity_filtered' 'lm_ca_wikiquote_filtered' 'lm_ca_wikipedia' 'lm_ca_wikimedia_filtered' 'lm_ca_wikisource_filtered' 'lm_ca_wikinews_filtered' 'lm_ca_wikibooks_filtered' 'lm_ar_wikisource_filtered' 'lm_en_wikipedia' 'lm_en_wikibooks_filtered' 'lm_en_wikiquote_filtered' 'lm_en_wikiversity_filtered' 'lm_en_wikivoyage_filtered' 'lm_en_wikinews_filtered' 'lm_es_wikipedia' 'lm_es_wikisource_filtered' 'lm_es_wikibooks_filtered' 'lm_es_wikivoyage_filtered' 'lm_es_wikinews_filtered' 'lm_es_wikiquote_filtered' 'lm_es_wikiversity_filtered' 'lm_es_wikihow_human_instructions' 'lm_eu_wikipedia' 'lm_eu_wikisource_filtered' 'lm_eu_wikibooks_filtered' 'lm_eu_wikiquote_filtered' 'lm_fr_wikisource_filtered' 'lm_fr_wikipedia' 'lm_fr_wikibooks_filtered' 'lm_fr_wikiversity_filtered' 'lm_fr_wikinews_filtered' 'lm_fr_wikivoyage_filtered' 'lm_fr_wikiquote_filtered' 'lm_fr_wikihow_human_instructions' 'lm_indic-mr_wikisource_filtered' 'lm_id_wikipedia' 'lm_id_wikisource_filtered' 'lm_id_wikibooks_filtered' 'lm_id_wikimedia_filtered' 'lm_id_wikiquote_filtered' 'lm_indic-as_wikisource_filtered' 'lm_indic-as_wikipedia' 'lm_indic-bn_wikisource_filtered' 'lm_indic-bn_wikipedia' 'lm_indic-bn_wikivoyage_filtered' 'lm_indic-bn_wikibooks_filtered' 'lm_indic-gu_wikisource_filtered' 'lm_indic-gu_wikipedia' 'lm_indic-gu_wikiquote_filtered' 'lm_indic-hi_wikisource_filtered' 'lm_indic-hi_wikipedia' 'lm_indic-hi_wikibooks_filtered' 'lm_indic-hi_wikiquote_filtered' 'lm_indic-hi_wikivoyage_filtered' 'lm_indic-hi_wikiversity_filtered' 'lm_indic-hi_wikimedia_filtered' 'lm_indic-hi_hindi_wikipedia_articles' 'lm_indic-kn_wikisource_filtered' 'lm_indic-kn_wikipedia' 'lm_indic-kn_wikiquote_filtered' 'lm_indic-ml_wikipedia' 'lm_indic-ml_wikisource_filtered' 'lm_indic-ml_wikiquote_filtered' 'lm_indic-ml_wikibooks_filtered' 'lm_indic-mr_wikipedia' 'lm_indic-mr_wikibooks_filtered' 'lm_indic-mr_wikiquote_filtered' 'lm_indic-or_wikipedia' 'lm_indic-or_wikisource_filtered' 'lm_indic-pa_wikipedia' 'lm_indic-pa_wikisource_filtered' 'lm_indic-pa_wikibooks_filtered' 'lm_indic-ta_wikisource_filtered' 'lm_indic-ta_wikipedia' 'lm_indic-ta_wikinews_filtered' 'lm_indic-ta_wikiquote_filtered' 'lm_indic-ta_wikibooks_filtered' 'lm_indic-te_wikipedia' 'lm_indic-te_wikisource_filtered' 'lm_indic-te_wikibooks_filtered' 'lm_indic-te_wikiquote_filtered' 'lm_indic-ur_wikipedia' 'lm_indic-ur_wikibooks_filtered' 'lm_indic-ur_wikiquote_filtered' 'lm_nigercongo-ak_wikipedia' 'lm_nigercongo-bm_wikibooks_filtered' 'lm_nigercongo-bm_wikipedia' 'lm_nigercongo-ig_wikipedia' 'lm_nigercongo-ki_wikipedia' 'lm_nigercongo-lg_wikipedia' 'lm_nigercongo-ln_wikipedia' 'lm_nigercongo-nso_wikipedia' 'lm_nigercongo-ny_wikipedia' 'lm_nigercongo-rn_wikipedia' 'lm_nigercongo-rw_wikipedia' 'lm_nigercongo-sn_wikipedia' 'lm_nigercongo-st_wikipedia' 'lm_nigercongo-sw_wikibooks_filtered' 'lm_nigercongo-sw_wikipedia' 'lm_nigercongo-tn_wikipedia' 'lm_nigercongo-ts_wikipedia' 'lm_nigercongo-tum_wikipedia' 'lm_nigercongo-tw_wikipedia' 'lm_nigercongo-wo_wikipedia' 'lm_nigercongo-wo_wikiquote_filtered' 'lm_nigercongo-yo_wikipedia' 'lm_nigercongo-zu_wikipedia' 'lm_pt_wikipedia' 'lm_pt_wikisource_filtered' 'lm_pt_wikinews_filtered' 'lm_pt_wikibooks_filtered' 'lm_pt_wikiversity_filtered' 'lm_pt_wikiquote_filtered' 'lm_pt_wikivoyage_filtered' 'lm_pt_wikimedia_filtered' 'lm_vi_wikipedia' 'lm_vi_wikisource_filtered' 'lm_vi_wikibooks_filtered' 'lm_vi_wikivoyage_filtered' 'lm_vi_wikiquote_filtered' 'lm_zh-cn_wikipedia' 'lm_zh_wikinews_filtered' 'lm_zh_wikibooks_filtered' 'lm_zh_wikiversity_filtered' 'lm_zh_wikivoyage_filtered' 'lm_zh_wikiquote_filtered' 'lm_zh-tw_wikipedia')

for ds_name in "${WIKI_DS[@]}"
do
echo -n "\n\n\n --------- $ds_name --------- \n" >> $DEBUG_LOG
python clean.py \
    --dataset-path bigscience-catalogue-lm-data/$ds_name \
    --maps-and-filters filter_wiki_user_titles filter_wiki_non_text_type \
    --save-path /home/lucile/data/result_filtering_cleaning/$ds_name.jsonl \
    --checks-save-path /home/lucile/data/result_filtering_cleaning/$ds_name_checks \
    --num-proc 40 \
    --sampling-size-map-check 100000 \
    --sampling-size-filter-check 100000 \
    --batch-size 100 \
    2>&1 | tee -a $DEBUG_LOG
done
