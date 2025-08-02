class VTC:
    voice_types_list = ['KCHI', 'CHI', 'FEM', 'MAL', 'SPEECH']

    # Define constants and file paths
    childlens_gt_df_file_path = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens/annotations_gt_id_split.pkl"

    rttm_og_01_file_path = "/home/nele_pauline_suffo/outputs/vtc/childlens_audio_test_01/all.rttm"
    rttm_og_2_file_path = "/home/nele_pauline_suffo/outputs/vtc/childlens_audio_test_2/all.rttm"
    output_og_01_file_path = "/home/nele_pauline_suffo/outputs/vtc/childlens_og_test_01.pkl"
    output_og_2_file_path = "/home/nele_pauline_suffo/outputs/vtc/childlens_og_test_2.pkl"

    rttm_ft_01_file_path = "/home/nele_pauline_suffo/outputs/vtc_finetuned/childlens_audio_test_01/all.rttm"
    rttm_ft_2_file_path = "/home/nele_pauline_suffo/outputs/vtc_finetuned/childlens_audio_test_2/all.rttm"
    output_ft_01_file_path = "/home/nele_pauline_suffo/outputs/vtc_finetuned/childlens_ft_test_01.pkl"
    output_ft_2_file_path = "/home/nele_pauline_suffo/outputs/vtc_finetuned/childlens_ft_test_2.pkl"

    rttm_cl_01_file_path = "/home/nele_pauline_suffo/outputs/vtc_from_scratch/childlens_audio_test_01/all.rttm"
    rttm_cl_2_file_path = "/home/nele_pauline_suffo/outputs/vtc_from_scratch/childlens_audio_test_2/all.rttm"
    output_cl_01_file_path = "/home/nele_pauline_suffo/outputs/vtc_from_scratch/childlens_cl_test_01.pkl"
    output_cl_2_file_path = "/home/nele_pauline_suffo/outputs/vtc_from_scratch/childlens_cl_test_2.pkl"