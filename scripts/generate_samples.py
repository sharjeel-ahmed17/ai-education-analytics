import os
import wave
import struct
import math
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def generate_samples(target_dir: str = "data/sample") -> None:
    """
    Generates sample files for all three modalities:
    1. Image: mock handwritten student worksheet (sample_worksheet.png)
    2. Time series: weekly student engagement metrics (sample_timeseries.csv)
    3. Audio: synthetic verbal response audio (sample_audio.wav)
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"Generating sample files in '{target_dir}'...")

    # 1. Generate sample_worksheet.png (Mock student exam worksheet)
    img_path = os.path.join(target_dir, "sample_worksheet.png")
    # Create a 256x256 image representing a worksheet page
    img = Image.new("RGB", (256, 256), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    # Draw simple gridlines to simulate a worksheet
    for y in range(20, 256, 40):
        draw.line([(0, y), (256, y)], fill=(200, 200, 200), width=1)
    # Draw some "handwritten" lines representing diagrams/answers
    draw.rectangle([40, 60, 100, 100], outline=(50, 50, 150), width=2) # Draw a square diagram
    draw.line([(40, 60), (100, 100)], fill=(50, 50, 150), width=2) # Draw diagonal line
    draw.text((120, 70), "x^2 + y^2 = r^2", fill=(30, 30, 100)) # mock handwritten formula
    draw.text((40, 150), "Q1: Legible math proof", fill=(30, 80, 30))
    # Save the image
    img.save(img_path)
    print(f"Saved: {img_path}")

    # 2. Generate sample_timeseries.csv (10 weeks of engagement + quiz grades)
    ts_path = os.path.join(target_dir, "sample_timeseries.csv")
    df = pd.DataFrame({
        "week": list(range(1, 11)),
        "lms_logins": [12, 14, 10, 8, 5, 4, 3, 2, 1, 2],
        "attendance_rate": [0.95, 0.98, 0.92, 0.90, 0.85, 0.82, 0.80, 0.75, 0.70, 0.68],
        "quiz_average": [88.0, 86.5, 85.0, 83.2, 79.5, 75.0, 72.8, 68.0, 64.5, 61.2]
    })
    df.to_csv(ts_path, index=False)
    print(f"Saved: {ts_path}")

    # 3. Generate sample_audio.wav (Synthetic 1-second 440Hz sine wave tone)
    audio_path = os.path.join(target_dir, "sample_audio.wav")
    sample_rate = 16000
    duration = 1.0 # seconds
    num_samples = int(sample_rate * duration)
    
    with wave.open(audio_path, "w") as w:
        w.setnchannels(1) # mono
        w.setsampwidth(2) # 16-bit
        w.setframerate(sample_rate)
        
        # Write sine wave values
        for i in range(num_samples):
            # 440Hz frequency
            value = int(16384 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            w.writeframes(struct.pack("<h", value))
            
    print(f"Saved: {audio_path}")
    print("All sample files generated successfully.")

if __name__ == "__main__":
    generate_samples()
