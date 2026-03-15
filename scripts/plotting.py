# =============================================================================
# plotting.py
# -----------------------------------------------------------------------------
# All Plotly visualizations for the project. Every function produces one
# interactive figure and saves it as both .html and .png to the given plot
# directory. Colour constants at the top of the file define the shared palette
# used across all charts.
#
# Functions:
#   plot_training_curves()        - Side-by-side line charts of train/val loss
#                                   and train/val accuracy across epochs.
#                                   Reads directly from train_stats.json.
#   plot_confusion_matrix()       - Normalised heatmap with raw sample counts
#                                   annotated in each cell.
#   plot_per_class_metrics()      - Grouped bar chart of precision, recall, and
#                                   F1 for each class, with support counts
#                                   shown below the bars.
#   plot_probability_distribution() - Violin plots showing the spread of the
#                                   model's max softmax confidence per
#                                   predicted class.
#   plot_example_predictions()    - Table of sampled predictions: for each
#                                   class, n random examples drawn from the
#                                   top-k most confident correct predictions
#                                   and top-k most confident incorrect
#                                   predictions.
#   plot_threshold_analysis()     - Dual line chart showing how per-class
#                                   accuracy and retained sample count change
#                                   as the confidence threshold increases from
#                                   0.0 to 1.0 in steps of 0.05.
# =============================================================================

import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import random
random.seed(45)


PRIMARY_COLOR = '#bf5700'
PRIMARY_ALT_COLOR = '#f7971f'
SECONDARY_COLOR = '#FFB06E'
ALT_1_COLOR = "#FFDDC2"
ALT_2_COLOR = '#BF0009'
ALT_3_COLOR = '#00BF57'
ALT_4_COLOR = '#FFFFFF'

label_colors = {'Normal':PRIMARY_COLOR,
                'Anxiety':PRIMARY_ALT_COLOR,
                'Depression':SECONDARY_COLOR,
                'Suicidal':ALT_1_COLOR}

def _save_fig(fig: go.Figure, plot_dir: str, name: str):
    os.makedirs(plot_dir, exist_ok=True)
    # fig.write_html(os.path.join(plot_dir, f"{name}.html"))
    fig.write_image(os.path.join(plot_dir, f"{name}.png"), width=1400, height=850, scale=2)


def plot_training_curves(train_stats_path: str, plot_dir: str):
    """Loss and accuracy curves from train_stats.json."""
    with open(train_stats_path) as f:
        stats = json.load(f)

    epochs = list(range(1, len(stats["train_epoch_losses"]) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss per Epoch", "Accuracy per Epoch"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Scatter(x=epochs, y=stats["train_epoch_losses"], mode="lines+markers",
                             name="Train Loss", line=dict(color=PRIMARY_COLOR, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=stats["val_epoch_losses"], mode="lines+markers",
                             name="Val Loss", line=dict(color=PRIMARY_ALT_COLOR, width=2, dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=stats["train_epoch_accs"], mode="lines+markers",
                             name="Train Acc", line=dict(color=PRIMARY_COLOR, width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=stats["val_epoch_accs"], mode="lines+markers",
                             name="Val Acc", line=dict(color=PRIMARY_ALT_COLOR, width=2, dash="dash"), showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="Epoch", dtick=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)
    fig.update_layout(
        title="Training Curves",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=0.45),
    )

    _save_fig(fig, plot_dir, "training_curves")
    return fig


def plot_confusion_matrix(all_labels, all_preds, label_names: dict, plot_dir: str, file_tag: str):
    """Normalized confusion matrix with raw counts annotated."""
    classes = sorted(label_names.keys())
    class_names = [label_names[c] for c in classes]
    cm = confusion_matrix(all_labels, all_preds, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annotations = [
        [f"{cm[i][j]}<br>({cm_norm[i][j]:.1%})" for j in range(len(classes))]
        for i in range(len(classes))
    ]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=13),
        colorscale="Oranges",
        zmin=0, zmax=1,
        colorbar=dict(title="Normalized"),
    ))
    fig.update_layout(
        title="Confusion Matrix (normalized, annotated with raw counts)",
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
    )

    _save_fig(fig, plot_dir, f"{file_tag}_confusion_matrix")
    return fig


def plot_per_class_metrics(class_report: dict, plot_dir: str, file_tag: str):
    """Grouped bar chart of precision, recall, and F1 per class."""
    skip = {"accuracy", "macro avg", "weighted avg"}
    classes = [k for k in class_report if k not in skip]
    metrics = ["precision", "recall", "f1-score"]
    colors = [PRIMARY_COLOR, PRIMARY_ALT_COLOR, ALT_4_COLOR]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=classes,
            y=[class_report[c][metric] for c in classes],
            marker_color=color,
            text=[f"{class_report[c][metric]:.3f}" for c in classes],
            textposition="outside",
        ))

    # Add support counts as a secondary y-axis annotation
    support = [class_report[c]["support"] for c in classes]
    fig.add_trace(go.Scatter(
        x=classes, y=[0] * len(classes),
        mode="text",
        text=[f"n={int(s)}" for s in support],
        textposition="bottom center",
        showlegend=False,
        textfont=dict(color="lightgrey", size=11),
    ))

    fig.update_layout(
        title="Per-Class Precision / Recall / F1",
        xaxis_title="Class",
        yaxis=dict(title="Score", range=[0, 1.12]),
        barmode="group",
        template="plotly_dark",
    )

    _save_fig(fig, plot_dir, f"{file_tag}_per_class_metrics")
    return fig


def plot_probability_distribution(all_preds, all_probs, label_names: dict, plot_dir: str, file_tag: str):
    """Violin plot of max softmax probability by predicted class."""
    fig = go.Figure()
    for label_idx, label_name in sorted(label_names.items()):
        probs_for_class = [p for pred, p in zip(all_preds, all_probs) if pred == label_idx]
        if probs_for_class:
            fig.add_trace(go.Violin(
                y=probs_for_class,
                name=label_name,
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                spanmode="hard",
                fillcolor=label_colors.get(label_name, None),
                line=dict(color='#000000')
            ))

    fig.update_layout(
        title="Prediction Confidence Distribution by Predicted Class",
        yaxis=dict(title="Max Softmax Probability", range=[0, 1]),
        xaxis_title="Predicted Class",
        template="plotly_white",
    )

    _save_fig(fig, plot_dir, f"{file_tag}_probability_distribution")
    return fig


def plot_example_predictions(
    texts: list,
    all_labels: list,
    all_preds: list,
    all_probs: list,
    label_names: dict,
    plot_dir: str,
    file_tag: str,
    n_per_class: int = 2,
    top_k: int = 100,
):
    """Table of example predictions: for each class, n_per_class random samples drawn
    from the top_k most confident correct predictions and top_k most confident incorrect
    predictions. Rows are grouped by class (correct block then incorrect block).
    """


    sample_idx = []
    for label_idx in sorted(label_names.keys()):
        # Top-k correct for this true label
        correct = [i for i in range(len(all_preds))
                   if all_labels[i] == label_idx and all_preds[i] == all_labels[i]]
        correct_pool = sorted(correct, key=lambda i: -all_probs[i])[:top_k]
        sample_idx += random.sample(correct_pool, min(n_per_class, len(correct_pool)))

    for label_idx in sorted(label_names.keys()):
        # Top-k incorrect for this true label
        incorrect = [i for i in range(len(all_preds))
                     if all_labels[i] == label_idx and all_preds[i] != all_labels[i]]
        incorrect_pool = sorted(incorrect, key=lambda i: -all_probs[i])[:top_k]
        sample_idx += random.sample(incorrect_pool, min(n_per_class, len(incorrect_pool)))

    def truncate(t, limit=130):
        return t[:limit] + "..." if len(t) > limit else t

    rows_text = [truncate(texts[i]) for i in sample_idx]
    rows_true = [label_names[all_labels[i]] for i in sample_idx]
    rows_pred = [label_names[all_preds[i]] for i in sample_idx]
    rows_prob = [f"{all_probs[i]:.1%}" for i in sample_idx]
    rows_flag = ["✓ Correct" if all_preds[i] == all_labels[i] else "✗ Wrong" for i in sample_idx]

    flag_colors = [
        ALT_3_COLOR if all_preds[i] == all_labels[i] else ALT_2_COLOR
        for i in sample_idx
    ]
    base_color = PRIMARY_COLOR
    text_color = '#FFFFFF'

    fig = go.Figure(go.Table(
        columnwidth=[4, 1, 1, 1, 1],
        header=dict(
            values=["<b>Text</b>", "<b>True Label</b>", "<b>Predicted</b>", "<b>Confidence</b>", "<b>Result</b>"],
            fill_color=PRIMARY_ALT_COLOR,
            align="left",
            font=dict(color="white", size=12),
            height=36,
        ),
        cells=dict(
            values=[rows_text, rows_true, rows_pred, rows_prob, rows_flag],
            fill_color=[
                [base_color] * len(sample_idx),
                [base_color] * len(sample_idx),
                [base_color] * len(sample_idx),
                [base_color] * len(sample_idx),
                [base_color] * len(sample_idx),
            ],
            align="left",
            font=dict(
                color=[
                [text_color] * len(sample_idx),
                [text_color] * len(sample_idx),
                [text_color] * len(sample_idx),
                [text_color] * len(sample_idx),
                flag_colors,
            ],
                size=11),
            height=32,
        ),
    ))
    fig.update_layout(
        title=f"Example Predictions — {n_per_class} random samples per class (correct & incorrect) from top-{top_k} by confidence",
        template="plotly_dark",
        height=max(500, 36 + 35 * len(sample_idx) + 120),
    )

    _save_fig(fig, plot_dir, f"{file_tag}_example_predictions")
    return fig


def plot_threshold_analysis(
    all_labels: list,
    all_preds: list,
    all_probs: list,
    label_names: dict,
    plot_dir: str,
    file_tag: str,
):
    """For each predicted class, plot accuracy and retained sample count as the
    confidence threshold increases from 0.0 to 1.0 in steps of 0.05.

    Accuracy at threshold t = fraction of predictions for that class with
    confidence >= t that match the true label (i.e. precision at threshold).
    Count at threshold t = number of such predictions.
    """
    thresholds = [round(t * 0.05, 2) for t in range(21)]  # 0.00 … 1.00

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy at Confidence Threshold", "Retained Sample Count at Threshold"),
        horizontal_spacing=0.12,
    )

    for label_idx, label_name in sorted(label_names.items()):
        color = label_colors.get(label_name, PRIMARY_COLOR)
        accs, counts = [], []

        for t in thresholds:
            idxs = [
                i for i in range(len(all_preds))
                if all_preds[i] == label_idx and all_probs[i] >= t
            ]
            counts.append(len(idxs))
            if idxs:
                correct = sum(1 for i in idxs if all_labels[i] == all_preds[i])
                accs.append(correct / len(idxs))
            else:
                accs.append(None)  # no predictions at this threshold — gap in line

        shared_kwargs = dict(
            x=thresholds,
            name=label_name,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=5),
            mode="lines+markers",
            legendgroup=label_name,
        )

        fig.add_trace(go.Scatter(y=accs, **shared_kwargs), row=1, col=1)
        fig.add_trace(go.Scatter(y=counts, showlegend=False, **shared_kwargs), row=1, col=2)

    fig.update_xaxes(title_text="Confidence Threshold", dtick=0.1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Sample Count", row=1, col=2)
    fig.update_layout(
        title="Per-Class Accuracy and Sample Retention vs. Confidence Threshold",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )

    _save_fig(fig, plot_dir, f"{file_tag}_threshold_analysis")
    return fig
