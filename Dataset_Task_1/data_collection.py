from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="SagarLekhak/SULAND_v2_RGB_Surface_Landmine_Dataset", filename="Annotation_files_for_SULAND_v2.zip", repo_type="dataset", local_dir="./datasets/")