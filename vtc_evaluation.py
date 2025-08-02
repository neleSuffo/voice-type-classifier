import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure, DetectionErrorRate
from pyannote.database.util import load_uem
from typing import List, Optional
import os
import argparse
from constants import VTC


HYPOTHESIS_PATHS = {
    'og_01': VTC.output_og_01_file_path,
    'og_2': VTC.output_og_2_file_path,
    'ft_01': VTC.output_ft_01_file_path,
    'ft_2': VTC.output_ft_2_file_path,
    'cl_01': VTC.output_cl_01_file_path,
    'cl_2': VTC.output_cl_2_file_path,
}

RTTM_PATHS = {
    'og_01': VTC.rttm_og_01_file_path,
    'og_2': VTC.rttm_og_2_file_path,
    'ft_01': VTC.rttm_ft_01_file_path,
    'ft_2': VTC.rttm_ft_2_file_path,
    'cl_01': VTC.rttm_cl_01_file_path,
    'cl_2': VTC.rttm_cl_2_file_path,
}


def plot_annotations_vs_predictions(audio_file_name: str, hypothesis_type: str, time_window: list, save_path: str = None) -> None:
    """
    Plot ground truth annotations and model predictions for a given audio file and time window.
    Optionally save the plot to a file.

    Parameters
    ----------
    audio_file_name : str
        The name of the audio file (without extension)
    hypothesis_type : str
        Hypothesis type: 'og_01', 'og_2', 'ft_01', 'ft_2', 'cl_01', 'cl_2'.
    time_window : list
        Start and end time for the plot: [start_time, end_time].
    save_path : str, optional
        If provided, save the plot to this path.
    """
    gt_path = VTC.childlens_gt_df_file_path
    gt_df = pd.read_pickle(gt_path)

    if hypothesis_type not in HYPOTHESIS_PATHS:
        raise ValueError("Invalid hypothesis type. Choose from the defined keys in HYPOTHESIS_PATHS.")

    hypothesis_df = pd.read_pickle(HYPOTHESIS_PATHS[hypothesis_type])
    hypothesis_df['Voice_type'] = hypothesis_df['Voice_type'].str.upper()
    
    # Accept both with and without extension, but prefer without
    gt = gt_df[gt_df['audio_file_name'].str.replace('.MP4', '').str.replace('.mp4', '') == audio_file_name]
    pred = hypothesis_df[hypothesis_df['audio_file_name'].str.replace('.MP4', '').str.replace('.mp4', '') == audio_file_name]

    num_categories = len(VTC.voice_types_list)
    fig_height_per_category = 0.5
    min_fig_height = 2.0
    figure_height = max(min_fig_height, num_categories * fig_height_per_category)
    
    fig, ax = plt.subplots(figsize=(10, figure_height))
    colors = {'GT': 'green', 'Pred': 'blue'}

    bar_height = 0.4
    row_step = 1.25
    gt_y_offset_in_category = 0.25
    pred_y_offset_in_category = -0.25

    yticks_positions = []
    yticklabels_text = []

    for i, vt in enumerate(VTC.voice_types_list[::-1]):
        category_center_y = i * row_step
        
        current_gt_bars = gt[gt['Voice_type'] == vt]
        for _, row in current_gt_bars.iterrows():
            ax.barh(
                y=category_center_y + gt_y_offset_in_category,
                width=row['Utterance_Duration'],
                left=row['Utterance_Start'],
                height=bar_height,
                color=colors['GT'],
                edgecolor='black',
                alpha=0.7
            )
        
        current_pred_bars = pred[pred['Voice_type'] == vt]
        for _, row in current_pred_bars.iterrows():
            ax.barh(
                y=category_center_y + pred_y_offset_in_category,
                width=row['Utterance_Duration'],
                left=row['Utterance_Start'],
                height=bar_height,
                color=colors['Pred'],
                edgecolor='black',
                alpha=0.7
            )
        
        yticks_positions.append(category_center_y)
        yticklabels_text.append(vt)

    ax.set_yticks(yticks_positions)
    ax.set_yticklabels(yticklabels_text)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Category")

    if num_categories > 0:
        min_bar_edge = (0 * row_step) + pred_y_offset_in_category - bar_height / 2
        max_bar_edge = ((num_categories - 1) * row_step) + gt_y_offset_in_category + bar_height / 2
        padding = 0.1 * row_step
        ax.set_ylim(min_bar_edge - padding, max_bar_edge + padding)
    else:
        ax.set_ylim(-0.5, 0.5)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=colors['GT'], alpha=0.7, label='Ground Truth'),
        mpatches.Patch(color=colors['Pred'], alpha=0.7, label='Prediction')
    ]
    ax.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    ax.set_xlim(time_window[0], time_window[1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
def combine_pickles(folder_path: str, output_file_name: str) -> None:
    """
    Combines all pickle files in a folder into a single DataFrame and saves it to a new pickle file.

    Parameters:
    - folder_path (str): Path to the folder containing the pickle files.
    - output_path (str): Path to save the combined DataFrame as a pickle file.
    """
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    output_path = os.path.join(folder_path, output_file_name)
    combined_df.to_pickle(output_path)
    
def rttm_to_dataframe(rttm_file: Path, output_path: Path) -> pd.DataFrame:
    """
    This function reads the voice_type_classifier
    output rttm file and returns its content as a pandas DataFrame.

    Parameters
    ----------
    rttm_file : path
        the path to the RTTM file
    output_path: path
        the path to the output pkl file

    """
    logging.info(f"Reading RTTM file from: {rttm_file}")
    
    try:
        df = pd.read_csv(
            rttm_file,
            sep=" ",
            names=[
                "Speaker",
                "audio_file_name",
                "audio_file_id",
                "Utterance_Start",
                "Utterance_Duration",
                "NA_1",
                "NA_2",
                "Voice_type",
                "NA_3",
                "NA_4",
            ],
        )
    except Exception as e:
        logging.error(f"Failed to read RTTM file: {e}")
        raise
    
    logging.info("Successfully read RTTM file. Processing data...")

    # Drop unnecessary columns
    df = df.drop(columns=["Speaker", "audio_file_id", "NA_1", "NA_2", "NA_3", "NA_4"])
    df["Utterance_End"] = df["Utterance_Start"] + df["Utterance_Duration"]
    
    logging.info("Data processing complete. Returning DataFrame.")

    try:
        df.to_pickle(output_path)
        logging.info(f"DataFrame successfully saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to file: {e}")
        raise

def dataframe_to_annotation(df, label_column="Voice_type"):
    """
    Converts a DataFrame to a pyannote.core.Annotation object.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'Utterance_Start', 'Utterance_End', and a label column.
    - label_column (str): Column name for the labels (default: 'Voice_type').

    Returns:
    - Annotation: pyannote.core.Annotation object.
    """
    annotation = Annotation()
    for _, row in df.iterrows():
        start = float(row["Utterance_Start"])
        end = float(row["Utterance_End"])
        label = row[label_column]
        annotation[Segment(start, end)] = label
    return annotation

def compute_metrics(hypothesis_type: str) -> None:
    """
    Computes and prints the detection metrics for the given hypothesis type.
    Parameters
    ----------
    hypothesis_type : str
        The type of hypothesis to compute metrics for. Options are 'og_01', 'og_2', 'ft_01', 'ft_2', 'cl_01', 'cl_2', 'cl_v3_01', or 'cl_v3_2'.
    """
    if hypothesis_type not in HYPOTHESIS_PATHS:
        raise ValueError(f"Invalid hypothesis type: '{hypothesis_type}'. Choose from {list(HYPOTHESIS_PATHS.keys())}.")
    
    hypothesis_path = HYPOTHESIS_PATHS[hypothesis_type]


    reference_path = VTC.childlens_gt_df_file_path
        
    try:
        hypothesis_df = pd.read_pickle(hypothesis_path)
        reference_df = pd.read_pickle(reference_path)
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Please ensure both reference and hypothesis pickle files exist. Details: {e}")
        return

    # Find common audio_file_names present in both reference and hypothesis
    reference_files_set = set(reference_df['audio_file_name'].unique())
    hypothesis_files_set = set(hypothesis_df['audio_file_name'].unique())
    
    annotated_files = list(reference_files_set.intersection(hypothesis_files_set))
    
    if not annotated_files:
        print(f"Warning: No common audio files found between reference and hypothesis for type '{hypothesis_type}'. Metrics cannot be computed.")
        return

    # Filter both DataFrames to only include these common files
    reference_df = reference_df[reference_df['audio_file_name'].isin(annotated_files)]
    predictions_df = hypothesis_df[hypothesis_df['audio_file_name'].isin(annotated_files)]
    
    uem_file_path = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens/complete.uem"
    try: 
        all_video_uems = load_uem(uem_file_path)
    except FileNotFoundError:
        print(f"Warning: UEM file not found at {uem_file_path}. UEM will be approximated.")
        all_video_uems = {}
    except Exception as e:
        print(f"Warning: Error loading UEM file {uem_file_path}: {e}. UEM will be approximated.")
        all_video_uems = {}

    metric_names = ['precision', 'recall', 'f1_score', 'error_rate']
    
    class_metrics_results = {
        vt: {metric_name: [] for metric_name in metric_names}
        for vt in VTC.voice_types_list
    }

    for video_filename in annotated_files:
        ref_video_df = reference_df[reference_df['audio_file_name'] == video_filename]
        pred_video_df = predictions_df[predictions_df['audio_file_name'] == video_filename]

        if ref_video_df.empty:
            print(f"Warning: No reference annotations found for video: {video_filename}, though it was in annotated_files. Skipping.")
            continue
        
        current_video_uem: Optional[Timeline] = all_video_uems.get(video_filename)
        
        for voice_type in VTC.voice_types_list:
            ref_class_video_df = ref_video_df[ref_video_df['Voice_type'] == voice_type]
            pred_class_video_df = pred_video_df[pred_video_df['Voice_type'] == voice_type]

            reference_annotation = dataframe_to_annotation(ref_class_video_df)
            hypothesis_annotation = dataframe_to_annotation(pred_class_video_df)

            detection_pr_f1_metric = DetectionPrecisionRecallFMeasure(collar=0, skip_overlap=False)
            detection_error_metric = DetectionErrorRate(collar=0, skip_overlap=False)

            _ = detection_pr_f1_metric(reference_annotation, hypothesis_annotation, uem=current_video_uem)
            p, r, f1 = detection_pr_f1_metric.compute_metrics()
            error_rate = detection_error_metric(reference_annotation, hypothesis_annotation, uem=current_video_uem)

            class_metrics_results[voice_type]['precision'].append(p)
            class_metrics_results[voice_type]['recall'].append(r)
            class_metrics_results[voice_type]['f1_score'].append(f1)
            class_metrics_results[voice_type]['error_rate'].append(error_rate)

    print(f"\nAveraged Metrics Per Class Over All Videos (Hypothesis: {hypothesis_type.upper()}):")
    
    overall_macro_f1_components = []

    for voice_type in VTC.voice_types_list:
        print(f"Class '{voice_type.upper()}':")
        for metric_name in metric_names:
            metric_values = class_metrics_results[voice_type][metric_name]
            
            valid_metric_values = [v for v in metric_values if not pd.isna(v)]
            
            if valid_metric_values:
                avg_metric = sum(valid_metric_values) / len(valid_metric_values)
            else:
                avg_metric = 0.0
            print(f"  Average {metric_name.replace('_', ' ').capitalize()}: {avg_metric:.3f}")
            
            if metric_name == 'f1_score':
                overall_macro_f1_components.append(avg_metric)
        print("")

    if overall_macro_f1_components:
        final_macro_f1_score = sum(overall_macro_f1_components) / len(overall_macro_f1_components)
    else:
        final_macro_f1_score = 0.0

    print(f"Final Macro F1 Score (average of per-class average F1 scores): {final_macro_f1_score:.3f}")

def compute_all_metrics_table() -> pd.DataFrame:
    """
    Computes detection metrics for all predefined hypothesis setups and returns them as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame where rows are hypothesis types and columns are metrics 
        (e.g., KCHI_precision, KCHI_recall, ..., Macro_F1_Score).
    """
    all_hypothesis_types = ['og_01', 'og_2', 'ft_01', 'ft_2', 'cl_01', 'cl_2',]
    
    hypothesis_paths_config = {
        'og_01': output_og_01_file_path,
        'og_2': output_og_2_file_path,
        'ft_01': output_ft_01_file_path,
        'ft_2': output_ft_2_file_path,
        'cl_01': output_cl_01_file_path,
        'cl_2': output_cl_2_file_path,
    }

    reference_df_global = pd.read_pickle(VTC.childlens_gt_df_file_path)

    uem_file_path = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens/complete.uem"
    try:
        all_video_uems_global = load_uem(uem_file_path)
    except FileNotFoundError:
        all_video_uems_global = {}
    except Exception:
        all_video_uems_global = {}

    metric_names = ['precision', 'recall', 'f1_score', 'error_rate']
    
    collected_results = []

    for hypothesis_type in all_hypothesis_types:
        hypothesis_path = hypothesis_paths_config[hypothesis_type]
        
        try:
            # First, check if the RTTM file exists and convert it
            rttm_file = RTTM_PATHS.get(hypothesis_type)
            if not os.path.exists(rttm_file):
                print(f"Warning: RTTM file not found for {hypothesis_type} at {rttm_file}. Skipping this hypothesis.")
                continue
                
            output_pkl_path = Path(hypothesis_path)
            rttm_to_dataframe(Path(rttm_file), output_pkl_path)
            
            # Then load the newly created pkl file
            hypothesis_df_original = pd.read_pickle(hypothesis_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: An error occurred processing {hypothesis_type}. Skipping. Details: {e}")
            row_data = {'Hypothesis': hypothesis_type}
            for vt in VTC.voice_types_list:
                for mn in metric_names:
                    row_data[f"{vt}_{mn}"] = np.nan
            row_data['Macro_F1_Score'] = np.nan
            collected_results.append(row_data)
            continue
        
        reference_df = reference_df_global.copy()
        hypothesis_df = hypothesis_df_original.copy()

        hypothesis_df['Voice_type'] = hypothesis_df['Voice_type'].str.upper().replace({'OCH': 'CHI'})

        reference_files_set = set(reference_df['audio_file_name'].unique())
        hypothesis_files_set = set(hypothesis_df['audio_file_name'].unique())
        annotated_files = list(reference_files_set.intersection(hypothesis_files_set))

        if not annotated_files:
            print(f"Warning: No common audio files found for '{hypothesis_type}'. Metrics will be NaN.")
            row_data = {'Hypothesis': hypothesis_type}
            for vt in VTC.voice_types_list:
                for mn in metric_names:
                    row_data[f"{vt}_{mn}"] = np.nan
            row_data['Macro_F1_Score'] = np.nan
            collected_results.append(row_data)
            continue

        reference_df = reference_df[reference_df['audio_file_name'].isin(annotated_files)]
        predictions_df = hypothesis_df[hypothesis_df['audio_file_name'].isin(annotated_files)]

        class_metrics_results = {
            vt: {metric_name: [] for metric_name in metric_names}
            for vt in VTC.voice_types_list
        }

        for video_filename in annotated_files:
            ref_video_df = reference_df[reference_df['audio_file_name'] == video_filename]
            pred_video_df = predictions_df[predictions_df['audio_file_name'] == video_filename]

            if ref_video_df.empty:
                continue
            
            current_video_uem: Optional[Timeline] = all_video_uems_global.get(video_filename)

            for voice_type in VTC.voice_types_list:
                ref_class_video_df = ref_video_df[ref_video_df['Voice_type'] == voice_type]
                pred_class_video_df = pred_video_df[pred_video_df['Voice_type'] == voice_type]

                reference_annotation = dataframe_to_annotation(ref_class_video_df)
                hypothesis_annotation = dataframe_to_annotation(pred_class_video_df)

                detection_pr_f1_metric = DetectionPrecisionRecallFMeasure(collar=0, skip_overlap=False)
                detection_error_metric = DetectionErrorRate(collar=0, skip_overlap=False)

                _ = detection_pr_f1_metric(reference_annotation, hypothesis_annotation, uem=current_video_uem)
                p, r, f1 = detection_pr_f1_metric.compute_metrics()
                error_rate = detection_error_metric(reference_annotation, hypothesis_annotation, uem=current_video_uem)

                class_metrics_results[voice_type]['precision'].append(p)
                class_metrics_results[voice_type]['recall'].append(r)
                class_metrics_results[voice_type]['f1_score'].append(f1)
                class_metrics_results[voice_type]['error_rate'].append(error_rate)
        
        current_hypothesis_metrics = {'Hypothesis': hypothesis_type}
        overall_macro_f1_components = []

        for voice_type in VTC.voice_types_list:
            for metric_name in metric_names:
                metric_values = class_metrics_results[voice_type][metric_name]
                valid_metric_values = [v for v in metric_values if not pd.isna(v)]
                
                avg_metric = np.nan
                if valid_metric_values:
                    avg_metric = sum(valid_metric_values) / len(valid_metric_values)
                else:
                    avg_metric = 0.0

                current_hypothesis_metrics[f"{voice_type}_{metric_name}"] = avg_metric
                
                if metric_name == 'f1_score' and not pd.isna(avg_metric):
                    overall_macro_f1_components.append(avg_metric)

        final_macro_f1_score = np.nan
        if overall_macro_f1_components:
            valid_f1_components = [f1 for f1 in overall_macro_f1_components if not pd.isna(f1)]
            if valid_f1_components:
                final_macro_f1_score = sum(valid_f1_components) / len(valid_f1_components)
            else:
                final_macro_f1_score = 0.0
        else:
            final_macro_f1_score = 0.0

        current_hypothesis_metrics['Macro_F1_Score'] = final_macro_f1_score
        collected_results.append(current_hypothesis_metrics)

    results_df = pd.DataFrame(collected_results)
    if not results_df.empty:
        results_df = results_df.set_index('Hypothesis')
        
        columns_ordered = []
        for vt in VTC.voice_types_list:
            for mn in metric_names:
                columns_ordered.append(f"{vt}_{mn}")
        columns_ordered.append('Macro_F1_Score')
        
        results_df = results_df.reindex(columns=[col for col in columns_ordered if col in results_df.columns])

    return results_df

def main():
    parser = argparse.ArgumentParser(description="Compute VTC evaluation metrics for a given hypothesis type or plot annotations vs predictions.")
    parser.add_argument("--hypothesis_type", type=str, help="The type of hypothesis to evaluate. E.g., 'og_01', 'og_2', 'ft_01', etc. Use 'all' to compute a table of all metrics.")
    parser.add_argument("--plot", action="store_true", help="If set, plot annotations vs predictions instead of computing metrics.")
    parser.add_argument("--video_name", type=str, help="The video name (without extension) to plot.")
    parser.add_argument("--time_window", type=str, help="Time window as start,end (e.g. 0,120)")
    args = parser.parse_args()

    # Pre-computation of all rttm files to pkl files
    for rttm_type, rttm_path in RTTM_PATHS.items():
        output_path = Path(HYPOTHESIS_PATHS[rttm_type])
        if Path(rttm_path).is_file():
            rttm_to_dataframe(Path(rttm_path), output_path)
        else:
            print(f"Warning: RTTM file not found for '{rttm_type}' at {rttm_path}. Skipping conversion.")

    if args.plot:
        if not args.video_name or not args.hypothesis_type or not args.time_window:
            print("Error: --plot requires --video_name, --hypothesis_type, and --time_window.")
            return
        time_window = [float(x) for x in args.time_window.split(",")]
        output_plot = f"{args.video_name}_{args.hypothesis_type}_annotations_vs_predictions.png"
        plot_annotations_vs_predictions(args.video_name, args.hypothesis_type, time_window, save_path=output_plot)
        return

    print("\n--- Evaluation ---")
    if args.hypothesis_type == "all":
        print("Computing metrics for all hypothesis types...")
        results_table = compute_all_metrics_table()
        if not results_table.empty:
            print(results_table.to_string())
        else:
            print("No results could be computed.")
    else:
        print(f"Computing metrics for '{args.hypothesis_type}'...")
        try:
            compute_metrics(args.hypothesis_type)
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()