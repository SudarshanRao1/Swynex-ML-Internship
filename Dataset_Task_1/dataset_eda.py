from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\SUDARSHAN\OneDrive\Documents\Swyenx-Internship\Dataset_Task_1\datasets"
)

OUTPUT_DIR = DATASET_ROOT / "EDA_annotation_results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# SULAND_v2 classes
CLASS_NAMES = {
    0: "PFM-1 (Butterfly)",
    1: "PMA-2 (Starfish)"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_split(path):

    path_string = str(path).lower()

    if "train" in path_string:
        return "Train"

    if "val" in path_string:
        return "Validation"

    if "test" in path_string:
        return "Test"

    return "Unknown"


def detect_domain(path):

    path_string = str(path).lower()

    if "data-iid" in path_string:
        return "IID"

    if "data-ood" in path_string:
        return "OOD"

    if "ita" in path_string:
        return "IID / Italy"

    if "usa" in path_string:
        return "OOD / USA"

    return "Unknown"


# ============================================================
# START
# ============================================================

print("\n" + "=" * 75)
print("SULAND_v2 ANNOTATION EXPLORATORY DATA ANALYSIS")
print("=" * 75)

print("\nDataset path:")
print(DATASET_ROOT)


if not DATASET_ROOT.exists():

    raise FileNotFoundError(
        f"\nDataset folder not found:\n{DATASET_ROOT}"
    )


# ============================================================
# FIND YOLO LABEL FILES
# ============================================================

label_files = list(
    DATASET_ROOT.rglob("*.txt")
)

print(
    f"\nYOLO annotation files found: "
    f"{len(label_files):,}"
)


# ============================================================
# READ ANNOTATIONS
# ============================================================

records = []

file_records = []

invalid_rows = 0


print("\nReading annotations...\n")


for label_path in tqdm(
    label_files,
    desc="Reading labels"
):

    split = detect_split(label_path)

    domain = detect_domain(label_path)

    object_count = 0


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


        for line_number, line in enumerate(
            lines,
            start=1
        ):

            parts = line.split()


            # YOLO format must have 5 values
            if len(parts) != 5:

                invalid_rows += 1

                continue


            try:

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])

                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                invalid_rows += 1

                continue


            # ------------------------------------------------
            # VALID CLASS
            # ------------------------------------------------

            if class_id not in CLASS_NAMES:

                invalid_rows += 1

                continue


            # ------------------------------------------------
            # VALID RANGE
            # ------------------------------------------------

            if not all(
                0 <= value <= 1
                for value in [
                    x_center,
                    y_center,
                    width,
                    height
                ]
            ):

                invalid_rows += 1

                continue


            if width <= 0 or height <= 0:

                invalid_rows += 1

                continue


            # ------------------------------------------------
            # BOUNDING BOX
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

                invalid_rows += 1

                continue


            # ------------------------------------------------
            # DERIVED FEATURES
            # ------------------------------------------------

            area = width * height

            aspect_ratio = (
                width / height
            )

            perimeter = (
                2 * (width + height)
            )


            # ------------------------------------------------
            # OBJECT SIZE
            # ------------------------------------------------

            if area < 0.01:

                object_size = "Small"

            elif area < 0.05:

                object_size = "Medium"

            else:

                object_size = "Large"


            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            records.append({

                "file":
                    str(label_path),

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

                "perimeter":
                    perimeter,

                "object_size":
                    object_size,

                "split":
                    split,

                "domain":
                    domain

            })


            object_count += 1


        # ----------------------------------------------------
        # FILE LEVEL INFORMATION
        # ----------------------------------------------------

        file_records.append({

            "file":
                str(label_path),

            "filename":
                label_path.name,

            "objects":
                object_count,

            "split":
                split,

            "domain":
                domain

        })


    except Exception:

        invalid_rows += 1


# ============================================================
# DATAFRAMES
# ============================================================

df = pd.DataFrame(records)

file_df = pd.DataFrame(file_records)


print(
    f"\nValid objects found: "
    f"{len(df):,}"
)

print(
    f"Invalid annotation rows: "
    f"{invalid_rows:,}"
)


if df.empty:

    raise RuntimeError(
        "No valid annotations were found."
    )


# ============================================================
# SAVE COMPLETE DATAFRAME
# ============================================================

df.to_csv(
    OUTPUT_DIR /
    "annotation_eda_dataset.csv",
    index=False
)

file_df.to_csv(
    OUTPUT_DIR /
    "file_level_statistics.csv",
    index=False
)


# ============================================================
# 1. BASIC SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("BASIC DATASET SUMMARY")
print("=" * 75)

print(
    f"Annotation files:       {len(label_files):,}"
)

print(
    f"Valid objects:          {len(df):,}"
)

print(
    f"Invalid rows:           {invalid_rows:,}"
)

print(
    f"Mean objects/file:      "
    f"{file_df['objects'].mean():.3f}"
)

print(
    f"Median objects/file:    "
    f"{file_df['objects'].median():.3f}"
)


# ============================================================
# 2. CLASS DISTRIBUTION
# ============================================================

class_counts = (
    df["class_name"]
    .value_counts()
)

class_percent = (
    df["class_name"]
    .value_counts(
        normalize=True
    ) * 100
)


class_summary = pd.DataFrame({

    "objects":
        class_counts,

    "percentage":
        class_percent

})


class_summary.to_csv(
    OUTPUT_DIR /
    "class_distribution.csv"
)


print("\nCLASS DISTRIBUTION")
print(class_summary)


plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=df,
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
    rotation=10
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "01_class_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. OBJECTS PER FILE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data=file_df,
    x="objects",
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
    "02_objects_per_file.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. BOUNDING BOX WIDTH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data=df,
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
    "03_bbox_width_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. BOUNDING BOX HEIGHT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data=df,
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
    "04_bbox_height_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. BOUNDING BOX AREA
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data=df,
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
    "05_bbox_area_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. LOG BOUNDING BOX AREA
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    np.log10(df["area"]),
    bins=50,
    kde=True
)

plt.title(
    "Log Bounding Box Area Distribution"
)

plt.xlabel(
    "log10(Bounding Box Area)"
)

plt.ylabel(
    "Number of Objects"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "06_log_bbox_area.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. ASPECT RATIO
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data=df,
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

# Remove extreme 1%
upper_limit = df[
    "aspect_ratio"
].quantile(0.99)

plt.xlim(
    0,
    upper_limit
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "07_bbox_aspect_ratio.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. OBJECT LOCATION
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.hexbin(
    df["x_center"],
    df["y_center"],
    gridsize=30,
    mincnt=1
)

plt.colorbar(
    label="Object Count"
)

plt.title(
    "Landmine Object Location Distribution"
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
    "08_object_location_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. CLASS VS AREA
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=df,
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
    rotation=10
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "09_class_vs_area.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. CLASS VS WIDTH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=df,
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
    rotation=10
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "10_class_vs_width.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. CLASS VS HEIGHT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=df,
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
    rotation=10
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "11_class_vs_height.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. OBJECT SIZE DISTRIBUTION
# ============================================================

size_order = [
    "Small",
    "Medium",
    "Large"
]


size_counts = (
    df["object_size"]
    .value_counts()
    .reindex(
        size_order,
        fill_value=0
    )
)


size_counts.to_csv(
    OUTPUT_DIR /
    "object_size_distribution.csv"
)


plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=df,
    x="object_size",
    order=size_order
)

plt.title(
    "Landmine Object Size Distribution"
)

plt.xlabel(
    "Object Size"
)

plt.ylabel(
    "Number of Objects"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "12_object_size_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. OBJECT SIZE BY CLASS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=df,
    x="object_size",
    hue="class_name",
    order=size_order
)

plt.title(
    "Object Size Distribution by Landmine Class"
)

plt.xlabel(
    "Object Size"
)

plt.ylabel(
    "Number of Objects"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "13_size_by_class.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. X/Y POSITION BY CLASS
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.scatterplot(
    data=df.sample(
        min(10000, len(df)),
        random_state=42
    ),
    x="x_center",
    y="y_center",
    hue="class_name",
    alpha=0.4
)

plt.title(
    "Landmine Spatial Distribution by Class"
)

plt.xlabel(
    "Normalized X"
)

plt.ylabel(
    "Normalized Y"
)

plt.xlim(0, 1)
plt.ylim(1, 0)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "14_spatial_distribution_by_class.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. CORRELATION MATRIX
# ============================================================

numeric_features = [

    "x_center",
    "y_center",
    "width",
    "height",
    "area",
    "aspect_ratio",
    "perimeter"

]


correlation = df[
    numeric_features
].corr()


correlation.to_csv(
    OUTPUT_DIR /
    "correlation_matrix.csv"
)


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Bounding Box Feature Correlation"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "15_correlation_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. IID / OOD ANALYSIS
# ============================================================

domain_counts = (
    df["domain"]
    .value_counts()
)


print("\nDOMAIN DISTRIBUTION")
print(domain_counts)


domain_counts.to_csv(
    OUTPUT_DIR /
    "domain_object_distribution.csv"
)


if len(domain_counts) > 1:

    plt.figure(
        figsize=(8, 5)
    )

    sns.countplot(
        data=df,
        x="domain"
    )

    plt.title(
        "IID vs OOD Object Distribution"
    )

    plt.xlabel(
        "Domain"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "16_iid_ood_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 18. DOMAIN VS OBJECT SIZE
# ============================================================

if df["domain"].nunique() > 1:

    plt.figure(
        figsize=(8, 5)
    )

    sns.countplot(
        data=df,
        x="domain",
        hue="object_size"
    )

    plt.title(
        "Object Size Distribution Across Domains"
    )

    plt.xlabel(
        "Domain"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "17_domain_vs_object_size.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 19. PCA
# ============================================================

print("\nRunning PCA...")

pca_features = [
    "x_center",
    "y_center",
    "width",
    "height",
    "area",
    "aspect_ratio"
]


X = df[
    pca_features
].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


pca_df = pd.DataFrame({

    "PC1": X_pca[:, 0],

    "PC2": X_pca[:, 1]

})


# Match class labels
pca_df["class"] = (
    df.loc[
        X.index,
        "class_name"
    ].values
)


pca_df.to_csv(
    OUTPUT_DIR /
    "PCA_features.csv",
    index=False
)


plt.figure(
    figsize=(9, 7)
)

sns.scatterplot(
    data=pca_df.sample(
        min(10000, len(pca_df)),
        random_state=42
    ),
    x="PC1",
    y="PC2",
    hue="class",
    alpha=0.5
)

plt.title(
    "PCA of Bounding Box Features"
)

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "18_PCA_bbox_features.png",
    dpi=300
)

plt.close()


print(
    "\nPCA explained variance:"
)

print(
    pca.explained_variance_ratio_
)


# ============================================================
# 20. K-MEANS CLUSTERING
# ============================================================

print("\nRunning K-Means clustering...")


cluster_features = [

    "width",
    "height",
    "area",
    "aspect_ratio"

]


cluster_data = df[
    cluster_features
].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()


cluster_scaler = StandardScaler()

cluster_scaled = (
    cluster_scaler.fit_transform(
        cluster_data
    )
)


# Test K values
k_values = range(2, 7)

silhouette_results = []


for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        cluster_scaled
    )

    score = silhouette_score(
        cluster_scaled,
        labels
    )

    silhouette_results.append({

        "k": k,

        "silhouette_score":
            score

    })


silhouette_df = pd.DataFrame(
    silhouette_results
)


silhouette_df.to_csv(
    OUTPUT_DIR /
    "kmeans_silhouette_scores.csv",
    index=False
)


best_k = int(
    silhouette_df.loc[
        silhouette_df[
            "silhouette_score"
        ].idxmax(),
        "k"
    ]
)


print(
    f"\nBest K according to silhouette score: "
    f"{best_k}"
)


# ============================================================
# FINAL K-MEANS
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)


cluster_labels = kmeans.fit_predict(
    cluster_scaled
)


cluster_result = cluster_data.copy()

cluster_result[
    "cluster"
] = cluster_labels


cluster_result.to_csv(
    OUTPUT_DIR /
    "kmeans_cluster_results.csv",
    index=False
)


# ============================================================
# CLUSTER VISUALIZATION
# ============================================================

cluster_pca = PCA(
    n_components=2
)


cluster_pca_data = cluster_pca.fit_transform(
    cluster_scaled
)


cluster_plot_df = pd.DataFrame({

    "PC1":
        cluster_pca_data[:, 0],

    "PC2":
        cluster_pca_data[:, 1],

    "cluster":
        cluster_labels

})


plt.figure(
    figsize=(9, 7)
)

sns.scatterplot(
    data=cluster_plot_df.sample(
        min(10000, len(cluster_plot_df)),
        random_state=42
    ),
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="tab10",
    alpha=0.5
)

plt.title(
    "K-Means Clustering of Bounding Box Features"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "19_kmeans_clusters.png",
    dpi=300
)

plt.close()


# ============================================================
# 21. CLUSTER DISTRIBUTION
# ============================================================

cluster_counts = (
    pd.Series(cluster_labels)
    .value_counts()
    .sort_index()
)


cluster_counts.to_csv(
    OUTPUT_DIR /
    "cluster_distribution.csv"
)


# ============================================================
# 22. TRAIN / VAL / TEST
# ============================================================

split_counts = (
    df["split"]
    .value_counts()
)


print("\nSPLIT DISTRIBUTION")
print(split_counts)


split_counts.to_csv(
    OUTPUT_DIR /
    "split_object_distribution.csv"
)


# ============================================================
# 23. FINAL STATISTICS TABLE
# ============================================================

statistics = {

    "Total annotation files":
        len(label_files),

    "Valid objects":
        len(df),

    "Invalid annotation rows":
        invalid_rows,

    "Mean objects per file":
        file_df["objects"].mean(),

    "Median objects per file":
        file_df["objects"].median(),

    "Mean bbox width":
        df["width"].mean(),

    "Mean bbox height":
        df["height"].mean(),

    "Mean bbox area":
        df["area"].mean(),

    "Median bbox area":
        df["area"].median(),

    "Mean bbox aspect ratio":
        df["aspect_ratio"].mean(),

    "Small objects":
        (df["object_size"] == "Small").sum(),

    "Medium objects":
        (df["object_size"] == "Medium").sum(),

    "Large objects":
        (df["object_size"] == "Large").sum(),

    "PFM-1 objects":
        (
            df["class_id"] == 0
        ).sum(),

    "PMA-2 objects":
        (
            df["class_id"] == 1
        ).sum()

}


statistics_df = pd.DataFrame(
    list(statistics.items()),
    columns=[
        "Metric",
        "Value"
    ]
)


statistics_df.to_csv(
    OUTPUT_DIR /
    "EDA_FINAL_STATISTICS.csv",
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 75)
print("EDA COMPLETED")
print("=" * 75)

print("\nResults saved at:")

print(
    OUTPUT_DIR
)

print("\nGenerated analyses:")

print("""
1. Class distribution
2. Objects per annotation file
3. Bounding-box width
4. Bounding-box height
5. Bounding-box area
6. Log bounding-box area
7. Bounding-box aspect ratio
8. Object location heatmap
9. Class vs bounding-box area
10. Class vs bounding-box width
11. Class vs bounding-box height
12. Object-size distribution
13. Object-size vs class
14. Spatial distribution by class
15. Correlation matrix
16. IID vs OOD distribution
17. Domain vs object size
18. PCA
19. K-Means clustering
20. Train/validation/test distribution
""")

print("\nYou can now inspect the CSV files and PNG plots.")