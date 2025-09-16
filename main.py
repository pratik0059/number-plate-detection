# ================================================
# Install dependencies (only first run)
# ================================================
!pip install easyocr

import cv2
import easyocr
import pandas as pd
from google.colab import files
import matplotlib.pyplot as plt
from datetime import datetime
import os
import re
from IPython.display import display

# ================================================
# Step 1: Choose Mode (no input() in Colab)
# ================================================
# Set manually: "1"=Images, "2"=Video, "3"=Both
mode = "1"   # <-- change here if needed

print("👉 Mode selected:", mode)

# ================================================
# Step 2: Upload Files
# ================================================
print("👉 Upload image(s) or video(s) depending on your choice")
uploaded = files.upload()
file_names = list(uploaded.keys())

# ================================================
# Step 3: Setup
# ================================================
reader = easyocr.Reader(['en'])

csv_filename = "detected_plates.csv"
plates_folder = "plates"

os.makedirs(plates_folder, exist_ok=True)

# Load existing CSV if exists
if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename)
    if "Plate_ID" in df.columns and pd.api.types.is_numeric_dtype(df["Plate_ID"]):
        last_id = df["Plate_ID"].max()
        if pd.isna(last_id):
            last_id = 0
    else:
        last_id = 0
else:
    df = pd.DataFrame(columns=["Plate_ID","Source","Full Plate",
                               "State Code","District Code","Series","Number",
                               "First Seen Time","Last Seen Time","Seen Count"])
    last_id = 0

plate_id_counter = int(last_id) + 1

# ================================================
# Step 4: Processing Function
# ================================================
def process_frame(frame, source_name):
    global df, plate_id_counter
    results = reader.readtext(frame)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plate_pattern = r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{3,4})$'

    for (bbox, text, prob) in results:
        clean_text = text.replace(" ", "").upper()

        # Match Indian plate pattern
        match = re.match(plate_pattern, clean_text)
        if match:
            state, district, series, number = match.groups()
        else:
            state, district, series, number = "", "", "", ""

        if clean_text not in df["Full Plate"].values:
            new_entry = {
                "Plate_ID": plate_id_counter,
                "Source": source_name,
                "Full Plate": clean_text,
                "State Code": state,
                "District Code": district,
                "Series": series,
                "Number": number,
                "First Seen Time": current_time,
                "Last Seen Time": current_time,
                "Seen Count": 1
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

            # Crop plate region
            x_min = int(min(pt[0] for pt in bbox))
            y_min = int(min(pt[1] for pt in bbox))
            x_max = int(max(pt[0] for pt in bbox))
            y_max = int(max(pt[1] for pt in bbox))

            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size > 0:
                cv2.imwrite(f"{plates_folder}/{clean_text}_{plate_id_counter}.jpg", cropped)

            plate_id_counter += 1
        else:
            df.loc[df["Full Plate"] == clean_text, "Seen Count"] += 1
            df.loc[df["Full Plate"] == clean_text, "Last Seen Time"] = current_time

        # Draw rectangle + text
        pts = [(int(x), int(y)) for (x, y) in bbox]
        cv2.rectangle(frame, pts[0], pts[2], (0,255,0), 2)
        cv2.putText(frame, clean_text, (pts[0][0], pts[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame

# ================================================
# Step 5: Process Uploaded Files
# ================================================
for file in file_names:
    if mode in ["1","3"] and file.lower().endswith((".jpg",".jpeg",".png")):
        img = cv2.imread(file)
        processed = process_frame(img, file)
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        plt.title(f"Processed Image: {file}")
        plt.axis("off")
        plt.show()

    elif mode in ["2","3"] and file.lower().endswith((".mp4",".avi",".mov",".mkv")):
        cap = cv2.VideoCapture(file)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_file = "processed_" + file
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            processed = process_frame(frame, file)
            out.write(processed)

            if frame_count % 150 == 0:
                plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                plt.title(f"Processed Frame {frame_count} from {file}")
                plt.axis("off")
                plt.show()

        cap.release()
        out.release()
        print(f"✅ Processed video saved as: {output_file}")
        files.download(output_file)

# ================================================
# Step 6: Save CSV
# ================================================
df.to_csv(csv_filename, index=False)
display(df)
files.download(csv_filename)
