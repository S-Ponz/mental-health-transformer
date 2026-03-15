import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix



PRIMARY_COLOR = '#bf5700'
PRIMARY_ALT_COLOR = '#f7971f'
SECONDARY_COLOR = '#FFB06E'
ALT_1_COLOR = '#BFB700'
ALT_2_COLOR = '#BF0009'
ALT_3_COLOR = '#00BF57'
ALT_4_COLOR = '#FFFFFF'

label_colors = {'Normal':PRIMARY_COLOR,
                'Anxiety':PRIMARY_ALT_COLOR,
                'Depression':SECONDARY_COLOR,
                'Suicidal':ALT_4_COLOR}

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
                fillcolor=label_colors.get(label_name, None)
            ))

    fig.update_layout(
        title="Prediction Confidence Distribution by Predicted Class",
        yaxis=dict(title="Max Softmax Probability", range=[0, 1]),
        xaxis_title="Predicted Class",
        template="plotly_dark",
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
    n: int = 16,
):
    """Table of example predictions: top-confidence correct and incorrect samples."""
    correct_idx = [i for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]
    incorrect_idx = [i for i in range(len(all_preds)) if all_preds[i] != all_labels[i]]

    n_correct = min(n // 2 + n % 2, len(correct_idx))
    n_incorrect = min(n // 2, len(incorrect_idx))

    # Sort by confidence descending to surface the most representative examples
    top_correct = sorted(correct_idx, key=lambda i: -all_probs[i])[:n_correct]
    top_incorrect = sorted(incorrect_idx, key=lambda i: -all_probs[i])[:n_incorrect]
    sample_idx = top_correct + top_incorrect

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
                size=13),
            height=32,
        ),
    ))
    fig.update_layout(
        title=f"Example Predictions — {n_correct} high-confidence correct + {n_incorrect} high-confidence incorrect",
        template="plotly_dark",
        height=max(500, 36 + 35 * len(sample_idx) + 120),
    )

    _save_fig(fig, plot_dir, f"{file_tag}_example_predictions")
    return fig
