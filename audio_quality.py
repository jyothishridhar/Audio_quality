import streamlit as st
import pandas as pd
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import io
import requests

# Function to extract samples from audio file
def extract_samples(audio_path):
    audio = AudioSegment.from_file(audio_path)
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
            plot_audio_with_issue(samples, dropout_position, "Audio_dropout", output_folder, frame_index, sample_rate)
            return f"Audio dropout detected at {dropout_position} samples", glitch_stats, audio_features

        # Check for clipping/distortion
        if audio_features['Max Amplitude'] >= 32767:
            clipping_position = np.argmax(np.abs(samples) >= 32000)
            plot_audio_with_issue(samples, clipping_position, "Audio_distortion", output_folder, frame_index, sample_rate)
            return f"Audio distortion detected at {clipping_position} samples", glitch_stats, audio_features

        # Check for consistent amplitude (glitches)
        amplitude_std = np.std(samples)
        if amplitude_std > 1000:
            glitch_position = np.argmax(samples)
            plot_audio_with_issue(samples, glitch_position, "Audio_glitch", output_folder, frame_index, sample_rate)

            # Calculate statistics for glitch values
            glitch_samples = samples[glitch_position:glitch_position + 1000]  # Adjust window size as needed
            glitch_stats['Mean'] = np.mean(glitch_samples)
            glitch_stats['Std'] = np.std(glitch_samples)

            return f"Audio glitch detected at {glitch_position} samples", glitch_stats, audio_features

        # If audio quality is good, plot the audio waveform
        plot_audio(samples, "Good_Audio_Quality", output_folder, frame_index, sample_rate)
        return "Audio quality is good", glitch_stats, audio_features

    except Exception as e:
        return f"Error: {str(e)}", glitch_stats, None

def plot_audio(samples, issue_label, output_folder, frame_index, sample_rate):
    os.makedirs(output_folder, exist_ok=True)

    time_values = np.arange(frame_index, frame_index + len(samples)) / sample_rate

    plt.figure(figsize=(16, 5))
    plt.plot(time_values, samples, label="Audio Signal", color='b')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"Audio_Waveform_{issue_label}_{frame_index}")

    # Save the plot to a file
    plot_filename = os.path.join(output_folder, f"audio_waveform_{issue_label}_{frame_index}.png")
    plt.savefig(plot_filename)
    plt.close()

    st.image(plot_filename, caption=f"Audio_Waveform_{issue_label}_{frame_index}", use_column_width=True)

    print(f"Plot saved to {plot_filename}")

def plot_audio_with_issue(samples, issue_position, issue_label, output_folder, frame_index, sample_rate):
    os.makedirs(output_folder, exist_ok=True)

    time_values = np.arange(frame_index, frame_index + len(samples)) / sample_rate

    plt.figure(figsize=(16, 5))
    plt.plot(time_values, samples, label="Audio Signal", color='b')
    plt.axvline(x=time_values[issue_position], color='r', linestyle='--', label=issue_label)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"Audio_Waveform_{issue_label}_{frame_index}")

    # Save the plot to a file
    plot_filename = os.path.join(output_folder, f"audio_waveform_{issue_label}_{frame_index}.png")
    plt.savefig(plot_filename)
    plt.close()

    st.image(plot_filename, caption=f"Audio_Waveform_{issue_label}_{frame_index}", use_column_width=True)

    print(f"Plot saved to {plot_filename}")

def download_audio(url):
    response = requests.get(url)
    return io.BytesIO(response.content)

# Streamlit app code
st.title("Audio Quality Assessment Demo")

# Git LFS URLs for the audio files
original_audio_url = "https://github.com/jyothishridhar/audio_quality/raw/main/referance_audio.wav"
distorted_audio_url = "https://github.com/jyothishridhar/audio_quality/raw/main/distorted_audio.wav"

# Download audio
original_audio_content = download_audio(original_audio_url)
distorted_audio_content = download_audio(distorted_audio_url)

# Add download links
st.markdown(f"**Download Original Audio**")
st.markdown(f"[Click here to download the Original Audio]({original_audio_url})")

st.markdown(f"**Download Distorted Audio**")
st.markdown(f"[Click here to download the Distorted Audio]({distorted_audio_url})")

if st.button("Run Audio Quality Assessment"):
    original_overlay_frames = detect_overlay(original_audio_content)
    distorted_overlay_frames = detect_overlay(distorted_audio_content)

    overlay_df, csv_report_path = generate_overlay_reports(original_overlay_frames, distorted_overlay_frames)

    # Display the result on the app
    st.success("Audio quality assessment completed! Result:")

    # Display the DataFrame
    st.dataframe(overlay_df)

    # Add download link for the report
    st.markdown(f"**Download Audio Quality Report**")
    st.markdown(f"[Click here to download the Audio Quality Report CSV]({csv_report_path})")
