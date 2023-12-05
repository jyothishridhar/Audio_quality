import os
import pandas as pd
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import io
import tempfile

# Function to extract samples from the audio file
def extract_samples(audio_content):
    audio = AudioSegment.from_file(io.BytesIO(audio_content), format="wav", codec="ffmpeg")
    return np.array(audio.get_array_of_samples())

# Function to calculate audio features
def calculate_audio_features(samples):
    features = {
        'Max Amplitude': np.max(np.abs(samples)),
        'Amplitude Std': np.std(samples),
    }
    return features

def evaluate_audio_quality_for_frame(samples, frame_index, frame_size, output_folder, sample_rate):
    try:
        # Initialize variables to store glitch stats and features
        glitch_stats = {'Mean': None, 'Std': None}
        audio_features = calculate_audio_features(samples)

        # Check for silence (dropouts)
        if audio_features['Max Amplitude'] < 0:
            dropout_position = np.argmax(samples < 0)
            plot_filename = plot_audio_with_issue(samples, dropout_position, "Audio_dropout", output_folder, frame_index, sample_rate)
            return f"Audio dropout detected at {dropout_position} samples", "Glitch", glitch_stats, audio_features, plot_filename

        # Check for clipping/distortion
        if audio_features['Max Amplitude'] >= 32767:
            clipping_position = np.argmax(np.abs(samples) >= 32000)
            plot_filename = plot_audio_with_issue(samples, clipping_position, "Audio_distortion", output_folder, frame_index, sample_rate)
            return f"Audio distortion detected at {clipping_position} samples", "Glitch", glitch_stats, audio_features, plot_filename

        # Check for consistent amplitude (glitches)
        amplitude_std = np.std(samples)
        if amplitude_std > 1000:
            glitch_position = np.argmax(samples)
            plot_filename = plot_audio_with_issue(samples, glitch_position, "Audio_glitch", output_folder, frame_index, sample_rate)

            # Calculate statistics for glitch values
            glitch_samples = samples[glitch_position:glitch_position + 1000]  # Adjust window size as needed
            glitch_stats['Mean'] = np.mean(glitch_samples)
            glitch_stats['Std'] = np.std(glitch_samples)

            return f"Audio glitch detected at {glitch_position} samples", "Glitch", glitch_stats, audio_features, plot_filename

        # If audio quality is good, plot the audio waveform
        plot_filename = plot_audio(samples, "Good_Audio_Quality", output_folder, frame_index, sample_rate)
        return "Audio quality is good", "No Glitch", glitch_stats, audio_features, plot_filename

    except Exception as e:
        return f"Error: {str(e)}", "Error", glitch_stats, None, None

def plot_audio(samples, issue_label, output_folder, frame_index, sample_rate):
    public_folder = "public_plots"
    os.makedirs(public_folder, exist_ok=True)

    time_values = np.arange(frame_index, frame_index + len(samples)) / sample_rate

    plt.figure(figsize=(16, 5))
    plt.plot(time_values, samples, label="Audio Signal", color='b')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"Audio_Waveform_{issue_label}_{frame_index}")

    # Save the plot to the public folder
    plot_filename = os.path.join(public_folder, f"audio_waveform_{issue_label}_{frame_index}.png")
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

def plot_audio_with_issue(samples, issue_position, issue_label, output_folder, frame_index, sample_rate):
    public_folder = "public_plots"
    os.makedirs(public_folder, exist_ok=True)

    time_values = np.arange(frame_index, frame_index + len(samples)) / sample_rate

    plt.figure(figsize=(16, 5))
    plt.plot(time_values, samples, label="Audio Signal", color='b')
    plt.axvline(x=time_values[issue_position], color='r', linestyle='--', label=issue_label)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"Audio_Waveform_{issue_label}_{frame_index}")

    # Save the plot to the public folder
    plot_filename = os.path.join(public_folder, f"audio_waveform_{issue_label}_{frame_index}.png")
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

# Streamlit app code
st.title("Audio Quality Analysis Demo")

# Git LFS URLs for the audio files
original_audio_url = "https://github.com/jyothishridhar/audio_quality/raw/main/referance_audio.wav"
distorted_audio_url = "https://github.com/jyothishridhar/audio_quality/raw/main/distorted_audio.wav"

# Download audio files
original_audio_content = requests.get(original_audio_url).content
distorted_audio_content = requests.get(distorted_audio_url).content

# Add download links
st.markdown(f"**Download Original Audio**")
st.markdown(f"[Click here to download the Original Audio]({original_audio_url})")

st.markdown(f"**Download Distorted Audio**")
st.markdown(f"[Click here to download the Distorted Audio]({distorted_audio_url})")

if st.button("Run Audio Quality Analysis"):
    original_samples = extract_samples(original_audio_content)
    distorted_samples = extract_samples(distorted_audio_content)

    # List to store results for each frame
    results_for_frames = []

    # Output folders
    original_output_folder = os.path.join(tempfile.gettempdir(), "original_plots")
    distorted_output_folder = os.path.join(tempfile.gettempdir(), "distorted_plots")

    # Sample rate
    sample_rate = 44100

    # Evaluate audio quality for each frame
    frame_size = int(sample_rate * 0.1)  # 100 milliseconds (adjust sample rate as needed)
    for i in range(0, min(len(original_samples), len(distorted_samples)), frame_size):
        original_frame_samples = original_samples[i:i + frame_size]
        distorted_frame_samples = distorted_samples[i:i + frame_size]

        result_original, glitch_status_original, glitch_stats_original, audio_features_original, plot_filename_original = evaluate_audio_quality_for_frame(
            original_frame_samples, i, frame_size, original_output_folder, sample_rate
        )

        result_distorted, glitch_status_distorted, glitch_stats_distorted, audio_features_distorted, plot_filename_distorted = evaluate_audio_quality_for_frame(
            distorted_frame_samples, i, frame_size, distorted_output_folder, sample_rate
        )

        results_for_frames.append({
            'Start Time (seconds)': i / sample_rate,  # Adjust the sample rate as needed
            'End Time (seconds)': (i + frame_size) / sample_rate,
            'Glitch Status (Original)': glitch_status_original,
            'Glitch Status (Distorted)': glitch_status_distorted,
            'Glitch Stats (Original)': glitch_stats_original,
            'Glitch Stats (Distorted)': glitch_stats_distorted,
            'Plot (Original)': plot_filename_original,
            'Plot (Distorted)': plot_filename_distorted,
        })

    # Create a DataFrame for the report
    report_df = pd.DataFrame(results_for_frames)

    # Save the report to a temporary file
    excel_file = os.path.join(tempfile.gettempdir(), "audio_quality_report_for_frames.xlsx")
    report_df.to_excel(excel_file, index=False)
    print(f"Report saved to {excel_file}")

    # Display the result on the app
    st.success("Audio quality analysis completed! Result:")

    # Display the DataFrame
    st.dataframe(report_df)

    # Display glitch plots
    st.markdown("### Glitch Plots (Original)")
    for i, plot_filename_original in enumerate(report_df['Plot (Original)']):
        st.image(plot_filename_original, f"Glitch Plot (Original) {i}")

    st.markdown("### Glitch Plots (Distorted)")
    for i, plot_filename_distorted in enumerate(report_df['Plot (Distorted)']):
        st.image(plot_filename_distorted, f"Glitch Plot (Distorted) {i}")

    # Add download link for the report
    st.markdown(f"**Download Audio Quality Report**")
    st.markdown(f"[Click here to download the Audio Quality Report Excel]({excel_file})")
