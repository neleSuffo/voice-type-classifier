from pathlib import Path
class VoiceTypeClassifier:
    CHILDLENS_CLASSES = ['KCHI', 'CHI', 'FEM', 'MAL', 'SPEECH']
    OUTPUT_DIR = Path("/home/nele_pauline_suffo/outputs")
    
    # Define constants and file paths
    CHILDLENS_GT_FILE_PATH = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens/annotations_gt_id_split.pkl"
    UEM_CHILDLENS_FILE_PATH = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens/complete.uem"
    
    VTC_OG_DIR = OUTPUT_DIR / "vtc"
    RTTM_OG_01_FILE_PATH = VTC_OG_DIR / "childlens_audio_test_01/all.rttm"
    RTTM_OG_2_FILE_PATH = VTC_OG_DIR / "childlens_audio_test_2/all.rttm"
    OUTPUT_OG_01_FILE_PATH = VTC_OG_DIR / "childlens_og_test_01.pkl"
    OUTPUT_OG_2_FILE_PATH = VTC_OG_DIR / "childlens_og_test_2.pkl"

    VTC_FINETUNED_DIR = OUTPUT_DIR / "vtc_finetuned"
    RTTM_FT_01_FILE_PATH = VTC_FINETUNED_DIR / "childlens_audio_test_01/all.rttm"
    RTTM_FT_2_FILE_PATH = VTC_FINETUNED_DIR / "childlens_audio_test_2/all.rttm"
    OUTPUT_FT_01_FILE_PATH = VTC_FINETUNED_DIR / "childlens_ft_test_01.pkl"
    OUTPUT_FT_2_FILE_PATH = VTC_FINETUNED_DIR / "childlens_ft_test_2.pkl"
    
    VTC_FROM_SCRATCH_DIR = OUTPUT_DIR / "vtc_from_scratch"
    RTTM_CL_01_FILE_PATH = "/home/nele_pauline_suffo/outputs/vtc_from_scratch/childlens_audio_test_01/all.rttm"
    RTTM_CL_2_FILE_PATH = "/home/nele_pauline_suffo/outputs/vtc_from_scratch/childlens_audio_test_2/all.rttm"
    OUTPUT_CL_01_FILE_PATH = "/home/nele_pauline_suffo/outputs/vtc_from_scratch/childlens_cl_test_01.pkl"
    OUTPUT_CL_2_FILE_PATH = "/home/nele_pauline_suffo/outputs/vtc_from_scratch/childlens_cl_test_2.pkl"
    
class AudioClassification:
    QUANTEX_CLASSES = ['KCHI', 'KCDS', 'OHS']
    QUANTEX_GT_FILE_PATH = "/home/nele_pauline_suffo/ProcessedData/audio_cls_input/annotations_gt_id_split.pkl"
    UEM_QUANTEX_FILE_PATH = "/home/nele_pauline_suffo/ProcessedData/audio_cls_input/complete.uem"
    
    RTTM_QUANTEX_01_FILE_PATH = "/home/nele_pauline_suffo/projects/voice-type-classifier/output_voice_type_classifier_qt/childlens_audio_three_classes_test/all.rttm"
    RTTM_QT_01_FILE_PATH = "/home/nele_pauline_suffo/projects/voice-type-classifier/output_voice_type_classifier_qt/childlens_audio_three_classes_test_01/all.rttm"
    RTTM_QT_2_FILE_PATH = "/home/nele_pauline_suffo/projects/voice-type-classifier/output_voice_type_classifier_qt/childlens_audio_three_classes_test_2/all.rttm"
    OUTPUT_QT_01_FILE_PATH = "/home/nele_pauline_suffo/outputs/audio_classification/vtc_three_classes_test_01.pkl"
    OUTPUT_QT_2_FILE_PATH = "/home/nele_pauline_suffo/outputs/audio_classification/vtc_three_classes_test_2.pkl"