from pathlib import Path
from collections import Counter, defaultdict
import hashlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

# Your annotation folder
DATASET_ROOT = Path(
    r"C:\Users\SUDARSHAN\OneDrive\Documents\Swyenx-Internship\Dataset_Task_1\datasets"
)

OUTPUT_DIR = DATASET_ROOT / "annotation_audit_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# SULAND_v2 classes
CLASS_NAMES = {
    0: "Butterfly (PFM-1)",
    1: "Starfish (PMA-2)"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_split(path):

    path_string = str(path).lower()

    if "train" in path_string:
        return "train"

    if "val" in path_string:
        return "val"

    if "test" in path_string:
        return "test"

    return "unknown"


def detect_domain(path):

    path_string = str(path).lower()

    if "data-iid" in path_string or "ita" in path_string:
        return "IID / Italy"

    if "data-ood" in path_string or "usa" in path_string:
        return "OOD / USA"

    return "unknown"


# ============================================================
# START
# ============================================================

print("\n" + "=" * 75)
print("SULAND_v2 YOLO ANNOTATION DATASET AUDIT")
print("=" * 75)

print("\nDataset:")
print(DATASET_ROOT)


if not DATASET_ROOT.exists():

    raise FileNotFoundError(
        f"\nDataset folder does not exist:\n{DATASET_ROOT}"
    )


# ============================================================
# FIND TXT FILES
# ============================================================

label_files = list(
    DATASET_ROOT.rglob("*.txt")
)


print(
    f"\nYOLO annotation files found: "
    f"{len(label_files):,}"
)


if len(label_files) == 0:

    raise RuntimeError(
        "No .txt annotation files found."
    )


# ============================================================
# STORAGE
# ============================================================

annotation_records = []

invalid_records = []

empty_files = []

file_statistics = []

class_counter = Counter()

split_counter = Counter()

domain_counter = Counter()


# ============================================================
# READ ALL ANNOTATIONS
# ============================================================

print("\nAnalyzing annotations...\n")


for label_path in tqdm(
    label_files,
    desc="Processing labels"
):

    split = detect_split(label_path)
    domain = detect_domain(label_path)

    split_counter[split] += 1
    domain_counter[domain] += 1

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]


        # ----------------------------------------------------
        # EMPTY LABEL
        # ----------------------------------------------------

        if len(lines) == 0:

            empty_files.append(
                str(label_path)
            )

            file_statistics.append({

                "file": str(label_path),
                "filename": label_path.name,
                "objects": 0,
                "split": split,
                "domain": domain

            })

            continue


        valid_objects = 0


        # ----------------------------------------------------
        # PROCESS EACH ANNOTATION
        # ----------------------------------------------------

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            parts = line.split()


            # ------------------------------------------------
            # CHECK NUMBER OF VALUES
            # ------------------------------------------------

            if len(parts) != 5:

                invalid_records.append({

                    "file": str(label_path),
                    "line": line_number,
                    "reason":
                        "Expected 5 values",
                    "content": line

                })

                continue


            # ------------------------------------------------
            # CONVERT VALUES
            # ------------------------------------------------

            try:

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])

                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                invalid_records.append({

                    "file": str(label_path),
                    "line": line_number,
                    "reason":
                        "Non-numeric value",
                    "content": line

                })

                continue


            # ------------------------------------------------
            # CLASS ID
            # ------------------------------------------------

            if class_id not in CLASS_NAMES:

                invalid_records.append({

                    "file": str(label_path),
                    "line": line_number,
                    "reason":
                        "Invalid class ID",
                    "class_id": class_id,
                    "content": line

                })

                continue


            # ------------------------------------------------
            # NORMALIZED COORDINATES
            # ------------------------------------------------

            values = [
                x_center,
                y_center,
                width,
                height
            ]


            if not all(
                0 <= value <= 1
                for value in values
            ):

                invalid_records.append({

                    "file": str(label_path),
                    "line": line_number,
                    "reason":
                        "Value outside [0,1]",
                    "content": line

                })

                continue


            # ------------------------------------------------
            # BOX SIZE
            # ------------------------------------------------

            if width <= 0:

                invalid_records.append({

                    "file": str(label_path),
                    "line": line_number,
                    "reason":
                        "Width <= 0",
                    "content": line

                })

                continue


            if height <= 0:

                invalid_records.append({

                    "file": str(label_path),
                    "line": line_number,
                    "reason":
                        "Height <= 0",
                    "content": line

                })

                continue


            # ------------------------------------------------
            # BOX BOUNDARY
            # ------------------------------------------------

            x1 = x_center - width / 2
            x2 = x_center + width / 2

            y1 = y_center - height / 2
            y2 = y_center + height / 2


            if (
                x1 < 0
                or x2 > 1
                or y1 < 0
                or y2 > 1
            ):

                invalid_records.append({

                    "file": str(label_path),
                    "line": line_number,
                    "reason":
                        "Bounding box outside image",
                    "content": line

                })

                continue


            # ------------------------------------------------
            # VALID OBJECT
            # ------------------------------------------------

            class_counter[class_id] += 1

            valid_objects += 1


            area = width * height

            aspect_ratio = (
                width / height
                if height > 0
                else np.nan
            )


            annotation_records.append({

                "file": str(label_path),

                "filename":
                    label_path.name,

                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[class_id],

                "x_center":
                    x_center,

                "y_center":
                    y_center,

                "width":
                    width,

                "height":
                    height,

                "area":
                    area,

                "aspect_ratio":
                    aspect_ratio,

                "split":
                    split,

                "domain":
                    domain

            })


        file_statistics.append({

            "file":
                str(label_path),

            "filename":
                label_path.name,

            "objects":
                valid_objects,

            "split":
                split,

            "domain":
                domain

        })


    except Exception as e:

        invalid_records.append({

            "file":
                str(label_path),

            "reason":
                f"File reading error: {e}"

        })


# ============================================================
# DATAFRAMES
# ============================================================

annotations_df = pd.DataFrame(
    annotation_records
)

invalid_df = pd.DataFrame(
    invalid_records
)

files_df = pd.DataFrame(
    file_statistics
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

annotations_df.to_csv(
    OUTPUT_DIR /
    "valid_annotations.csv",
    index=False
)

invalid_df.to_csv(
    OUTPUT_DIR /
    "invalid_annotations.csv",
    index=False
)

files_df.to_csv(
    OUTPUT_DIR /
    "file_statistics.csv",
    index=False
)

pd.DataFrame({
    "empty_annotation_file":
        empty_files
}).to_csv(
    OUTPUT_DIR /
    "empty_annotation_files.csv",
    index=False
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_distribution = []

total_objects = sum(
    class_counter.values()
)


for class_id in sorted(
    CLASS_NAMES.keys()
):

    count = class_counter[class_id]

    percentage = (
        count / total_objects * 100
        if total_objects > 0
        else 0
    )

    class_distribution.append({

        "class_id":
            class_id,

        "class_name":
            CLASS_NAMES[class_id],

        "objects":
            count,

        "percentage":
            percentage

    })


class_df = pd.DataFrame(
    class_distribution
)


class_df.to_csv(
    OUTPUT_DIR /
    "class_distribution.csv",
    index=False
)


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

print("\n" + "-" * 75)
print("CLASS DISTRIBUTION")
print("-" * 75)

for _, row in class_df.iterrows():

    print(
        f"{row['class_id']} - "
        f"{row['class_name']:<25} "
        f"{int(row['objects']):>7,} objects "
        f"({row['percentage']:.2f}%)"
    )


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

print("\n" + "-" * 75)
print("SPLIT DISTRIBUTION")
print("-" * 75)

split_df = pd.DataFrame(
    split_counter.items(),
    columns=[
        "split",
        "annotation_files"
    ]
)

print(split_df.to_string(index=False))

split_df.to_csv(
    OUTPUT_DIR /
    "split_distribution.csv",
    index=False
)


# ============================================================
# DOMAIN DISTRIBUTION
# ============================================================

print("\n" + "-" * 75)
print("DOMAIN DISTRIBUTION")
print("-" * 75)

domain_df = pd.DataFrame(
    domain_counter.items(),
    columns=[
        "domain",
        "annotation_files"
    ]
)

print(domain_df.to_string(index=False))

domain_df.to_csv(
    OUTPUT_DIR /
    "domain_distribution.csv",
    index=False
)


# ============================================================
# INVALID ANNOTATION SUMMARY
# ============================================================

print("\n" + "-" * 75)
print("ANNOTATION QUALITY")
print("-" * 75)

print(
    f"Total annotation files:     "
    f"{len(label_files):,}"
)

print(
    f"Valid objects:              "
    f"{len(annotations_df):,}"
)

print(
    f"Invalid annotation rows:    "
    f"{len(invalid_df):,}"
)

print(
    f"Empty annotation files:     "
    f"{len(empty_files):,}"
)


if len(invalid_df) > 0:

    print("\nInvalid annotation reasons:")

    print(
        invalid_df["reason"]
        .value_counts()
        .to_string()
    )


# ============================================================
# OBJECTS PER FILE
# ============================================================

if len(files_df) > 0:

    print("\n" + "-" * 75)
    print("OBJECTS PER ANNOTATION FILE")
    print("-" * 75)

    print(
        files_df["objects"].describe()
    )


# ============================================================
# BOUNDING BOX STATISTICS
# ============================================================

if len(annotations_df) > 0:

    print("\n" + "-" * 75)
    print("BOUNDING BOX STATISTICS")
    print("-" * 75)

    bbox_stats = annotations_df[
        [
            "width",
            "height",
            "area",
            "aspect_ratio"
        ]
    ].describe()

    print(
        bbox_stats
    )

    bbox_stats.to_csv(
        OUTPUT_DIR /
        "bounding_box_statistics.csv"
    )


# ============================================================
# EDA GRAPH 1
# CLASS DISTRIBUTION
# ============================================================

if len(annotations_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )

    sns.countplot(
        data=annotations_df,
        x="class_name"
    )

    plt.title(
        "SULAND_v2 Landmine Class Distribution"
    )

    plt.xlabel(
        "Landmine Class"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "01_class_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EDA GRAPH 2
# BOUNDING BOX WIDTH
# ============================================================

if len(annotations_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=annotations_df,
        x="width",
        bins=50,
        kde=True
    )

    plt.title(
        "Bounding Box Width Distribution"
    )

    plt.xlabel(
        "Normalized Width"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "02_bbox_width_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EDA GRAPH 3
# BOUNDING BOX HEIGHT
# ============================================================

if len(annotations_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=annotations_df,
        x="height",
        bins=50,
        kde=True
    )

    plt.title(
        "Bounding Box Height Distribution"
    )

    plt.xlabel(
        "Normalized Height"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "03_bbox_height_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EDA GRAPH 4
# BOUNDING BOX AREA
# ============================================================

if len(annotations_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=annotations_df,
        x="area",
        bins=50,
        kde=True
    )

    plt.title(
        "Bounding Box Area Distribution"
    )

    plt.xlabel(
        "Normalized Bounding Box Area"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "04_bbox_area_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EDA GRAPH 5
# BOUNDING BOX ASPECT RATIO
# ============================================================

if len(annotations_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=annotations_df,
        x="aspect_ratio",
        bins=50,
        kde=True
    )

    plt.title(
        "Bounding Box Aspect Ratio"
    )

    plt.xlabel(
        "Width / Height"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.xlim(
        0,
        annotations_df["aspect_ratio"].quantile(0.99)
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "05_bbox_aspect_ratio.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EDA GRAPH 6
# OBJECT LOCATION HEATMAP
# ============================================================

if len(annotations_df) > 0:

    plt.figure(
        figsize=(8, 6)
    )

    plt.hexbin(
        annotations_df["x_center"],
        annotations_df["y_center"],
        gridsize=30,
        mincnt=1
    )

    plt.colorbar(
        label="Object Count"
    )

    plt.title(
        "Landmine Bounding Box Center Distribution"
    )

    plt.xlabel(
        "Normalized X Position"
    )

    plt.ylabel(
        "Normalized Y Position"
    )

    plt.xlim(0, 1)
    plt.ylim(1, 0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "06_object_location_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EDA GRAPH 7
# CLASS VS AREA
# ============================================================

if len(annotations_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )

    sns.boxplot(
        data=annotations_df,
        x="class_name",
        y="area"
    )

    plt.title(
        "Bounding Box Area by Landmine Class"
    )

    plt.xlabel(
        "Landmine Class"
    )

    plt.ylabel(
        "Normalized Bounding Box Area"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "07_class_vs_bbox_area.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EDA GRAPH 8
# CLASS VS WIDTH
# ============================================================

if len(annotations_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )

    sns.boxplot(
        data=annotations_df,
        x="class_name",
        y="width"
    )

    plt.title(
        "Bounding Box Width by Landmine Class"
    )

    plt.xlabel(
        "Landmine Class"
    )

    plt.ylabel(
        "Normalized Width"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "08_class_vs_bbox_width.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EDA GRAPH 9
# CLASS VS HEIGHT
# ============================================================

if len(annotations_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )

    sns.boxplot(
        data=annotations_df,
        x="class_name",
        y="height"
    )

    plt.title(
        "Bounding Box Height by Landmine Class"
    )

    plt.xlabel(
        "Landmine Class"
    )

    plt.ylabel(
        "Normalized Height"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "09_class_vs_bbox_height.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EDA GRAPH 10
# OBJECTS PER FILE
# ============================================================

if len(files_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        files_df["objects"],
        bins=20
    )

    plt.title(
        "Objects per Annotation File"
    )

    plt.xlabel(
        "Number of Objects"
    )

    plt.ylabel(
        "Number of Annotation Files"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "10_objects_per_file.png",
        dpi=300
    )

    plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "Total annotation files":
        len(label_files),

    "Total valid objects":
        len(annotations_df),

    "Invalid annotation rows":
        len(invalid_df),

    "Empty annotation files":
        len(empty_files),

    "PFM-1 objects":
        class_counter[0],

    "PMA-2 objects":
        class_counter[1]

}


summary_df = pd.DataFrame(
    list(summary.items()),
    columns=[
        "Metric",
        "Value"
    ]
)


summary_df.to_csv(
    OUTPUT_DIR /
    "FINAL_SUMMARY.csv",
    index=False
)


# ============================================================
# FINAL PRINT
# ============================================================

print("\n")
print("=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

for key, value in summary.items():

    print(
        f"{key:<35}: {value:,}"
    )


print("\nResults saved to:")

print(
    OUTPUT_DIR
)

print("\nAudit completed successfully.")