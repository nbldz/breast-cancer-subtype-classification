"""
interface/visualisation.py
CDLS Six-Panel Research Visualisation Interface (Fig. 2 / Section III-P).

Panels:
  P1: Scenario input and trajectory launch (patient ID, scenario selection, z0 display)
  P2: WSI attention heatmap with step-wise playback
  P3: UMAP trajectory vs. real patient manifold (τ=0 → τ=4)
  P4: Per-step entropy, probability, and response-score curves
  P5: Baseline vs. scenario-conditioned PAM50 probability shift
  P6: kNN patient retrieval with latent-retrieval survival statistics

IMPORTANT (Section III-P):
  - No formal user study is conducted.
  - This is a RESEARCH VISUALISATION TOOL ONLY, not a clinical decision-support system.
  - Informal review by 3 domain researchers; limitations: n=3, no structured protocol.
  - Trajectory panel (P3) and patient retrieval panel (P6) identified as most informative.

Usage:
    python interface/visualisation.py \\
        --checkpoint checkpoints/seed_789/ppo_best_seed789.pt \\
        --patient_id TCGA-3C-AAAU \\
        --scenario 0

    # Or launch interactive mode:
    python interface/visualisation.py --interactive
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # Change to "TkAgg" or "Qt5Agg" for interactive display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cdls  import CDLS
from models.feedback import CosineSimilarityRetrieval

# PAM50 subtype colours
SUBTYPE_COLOURS = {
    0: "#E74C3C",   # Basal  – red
    1: "#3498DB",   # HER2   – blue
    2: "#2ECC71",   # LumA   – green
    3: "#F39C12",   # LumB   – orange
    4: "#9B59B6",   # Normal – purple
}
SUBTYPE_NAMES = {0: "Basal", 1: "HER2", 2: "LumA", 3: "LumB", 4: "Normal"}

SCENARIO_NAMES = {
    0: "High HER2",
    1: "High LumA",
    2: "Hormone Therapy",
}


class CDLSVisualiser:
    """
    Six-panel research visualisation tool for CDLS.
    """

    def __init__(
        self,
        model:     CDLS,
        device:    torch.device,
        z_bank:    Optional[torch.Tensor] = None,     # (N_train, 7) all training embeddings
        labels_bank: Optional[np.ndarray] = None,     # (N_train,)  subtype labels
        patient_ids_bank: Optional[List[str]] = None,
        surv_bank: Optional[np.ndarray] = None,       # (N_train,) survival times
    ):
        self.model   = model.eval().to(device)
        self.device  = device
        self.z_bank  = z_bank
        self.labels_bank      = labels_bank
        self.patient_ids_bank = patient_ids_bank
        self.surv_bank        = surv_bank

        # Fit UMAP on training bank (lazy)
        self._umap_reducer = None
        self._umap_z_train = None

    # ─────────────────────────────────────────────────────────────────────────

    def _fit_umap(self):
        """Fit UMAP reducer on training bank embeddings."""
        if self._umap_reducer is not None:
            return
        try:
            import umap
            print("Fitting UMAP on training bank...")
            self._umap_reducer = umap.UMAP(
                n_components=2, n_neighbors=15, min_dist=0.1,
                metric="euclidean", random_state=789,
            )
            z_np = self.z_bank.numpy() if isinstance(self.z_bank, torch.Tensor) \
                   else self.z_bank
            self._umap_z_train = self._umap_reducer.fit_transform(z_np)
        except ImportError:
            print("WARNING: umap-learn not installed. P3 panel unavailable.")
            self._umap_reducer = None

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run_patient(
        self,
        batch:        Dict,
        scenario_idx: Optional[int] = None,
        return_all:   bool = True,
    ) -> Dict:
        """
        Run full trajectory for a single patient batch.
        Returns rich output for all 6 panels.
        """
        # Encode
        z0, attn_weights = self.model.encode(batch)

        # Baseline trajectory (no scenario)
        base_out  = self.model.rollout(z0, return_trajectory=True)
        base_traj = base_out["trajectory"]                     # list of (B, 7)
        base_proba = [
            torch.softmax(self.model.classifier(z), dim=-1).cpu().numpy()
            for z in base_traj
        ]
        base_entropy = self.model.get_per_step_entropy(base_traj)

        # Scenario trajectory (if requested)
        scen_traj  = None
        scen_proba = None
        if scenario_idx is not None:
            scen_out   = self.model.rollout(z0, scenario_idx=scenario_idx,
                                             return_trajectory=True)
            scen_traj  = scen_out["trajectory"]
            scen_proba = [
                torch.softmax(self.model.classifier(z), dim=-1).cpu().numpy()
                for z in scen_traj
            ]

        # kNN retrieval (P6) — cosine similarity (Eq. 13)
        z_T  = base_traj[-1]
        knn_indices   = None
        knn_distances = None
        if self.z_bank is not None:
            retrieval = CosineSimilarityRetrieval(self.z_bank)
            retrieval.set_bank(self.z_bank)
            knn_indices, knn_sims = retrieval.retrieve(z_T.to("cpu"), k=5)
            knn_indices   = knn_indices[0].tolist()   # first patient
            knn_distances = knn_sims[0].tolist()

        return {
            "z0":          z0.cpu(),
            "attn_weights": attn_weights.cpu(),
            "base_traj":   [z.cpu() for z in base_traj],
            "base_proba":  base_proba,
            "base_entropy": base_entropy,
            "scen_traj":   [z.cpu() for z in scen_traj] if scen_traj else None,
            "scen_proba":  scen_proba,
            "knn_indices":    knn_indices,
            "knn_distances":  knn_distances,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def plot_six_panels(
        self,
        patient_result: Dict,
        patient_id:     str = "TCGA-??-????",
        scenario_idx:   Optional[int] = None,
        wsi_patch:      Optional[np.ndarray] = None,   # (H, W, 3) uint8
        save_path:      Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate the six-panel research visualisation (Fig. 2).
        """
        fig = plt.figure(figsize=(22, 14))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        ax_p1 = fig.add_subplot(gs[0, 0])
        ax_p2 = fig.add_subplot(gs[0, 1])
        ax_p3 = fig.add_subplot(gs[0, 2])
        ax_p4 = fig.add_subplot(gs[1, 0])
        ax_p5 = fig.add_subplot(gs[1, 1])
        ax_p6 = fig.add_subplot(gs[1, 2])

        res = patient_result
        T   = len(res["base_traj"]) - 1
        tau_steps = list(range(T + 1))

        # ── P1: Trajectory Configuration ─────────────────────────────────
        ax_p1.axis("off")
        z0_np = res["z0"][0].numpy()
        info_text = (
            f"Patient ID: {patient_id}\n\n"
            f"Scenario: {SCENARIO_NAMES.get(scenario_idx, 'None')}\n\n"
            f"Latent z0:\n"
            + "  [" + ", ".join(f"{v:.3f}" for v in z0_np) + "]\n\n"
            f"Refinement steps T={T}\n"
            f"Feedback λ={self.model.feedback.lambda_feedback}, "
            f"k={self.model.feedback.k}\n\n"
            f"NOTE: Research tool only.\n"
            f"Not for clinical use."
        )
        ax_p1.text(0.05, 0.95, info_text, transform=ax_p1.transAxes,
                   fontsize=9, verticalalignment="top", fontfamily="monospace",
                   bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax_p1.set_title("P1: Scenario Input & Trajectory Config", fontsize=10, fontweight="bold")

        # ── P2: WSI Attention Heatmap ─────────────────────────────────────
        ax_p2.axis("off")
        attn = res["attn_weights"][0].numpy()  # (N_patches,)
        if wsi_patch is not None:
            im = ax_p2.imshow(wsi_patch)
            ax_p2.set_title(
                f"P2: WSI Attention (τ=0→{T})\nAttention visualisation only",
                fontsize=10, fontweight="bold"
            )
        else:
            # Show attention as bar plot (no slide available)
            n_show = min(50, len(attn))
            ax_p2.barh(range(n_show), sorted(attn, reverse=True)[:n_show],
                       color="steelblue", alpha=0.7)
            ax_p2.set_xlabel("Attention Weight")
            ax_p2.set_ylabel("Patch rank")
            ax_p2.set_title("P2: WSI Attention Weights\n(Attention visualisation only)",
                             fontsize=10, fontweight="bold")

        # ── P3: UMAP Trajectory ──────────────────────────────────────────
        self._fit_umap()
        ax_p3.set_title(
            f"P3: UMAP Trajectory (τ=0→{T})\nvs. Training Patient Manifold",
            fontsize=10, fontweight="bold"
        )
        if self._umap_reducer is not None and self._umap_z_train is not None:
            # Background: training manifold
            if self.labels_bank is not None:
                for c in range(5):
                    mask = self.labels_bank == c
                    ax_p3.scatter(
                        self._umap_z_train[mask, 0],
                        self._umap_z_train[mask, 1],
                        c=SUBTYPE_COLOURS[c], alpha=0.15, s=8, label=SUBTYPE_NAMES[c]
                    )

            # Project patient trajectory
            base_zs = torch.cat([z[0:1] for z in res["base_traj"]], dim=0).numpy()
            traj_2d = self._umap_reducer.transform(base_zs)

            ax_p3.plot(traj_2d[:, 0], traj_2d[:, 1], "k-o",
                       markersize=8, zorder=5, label="Trajectory")
            for t_idx, (x, y) in enumerate(traj_2d):
                ax_p3.annotate(f"τ={t_idx}", (x, y),
                               fontsize=7, ha="left", va="bottom")

            if res["scen_traj"] is not None:
                scen_zs  = torch.cat([z[0:1] for z in res["scen_traj"]], dim=0).numpy()
                scen_2d  = self._umap_reducer.transform(scen_zs)
                ax_p3.plot(scen_2d[:, 0], scen_2d[:, 1], "r--o",
                           markersize=6, zorder=5, alpha=0.8, label="Scenario")

            ax_p3.legend(fontsize=7, loc="best")
            ax_p3.set_xlabel("UMAP-1", fontsize=9)
            ax_p3.set_ylabel("UMAP-2", fontsize=9)
        else:
            ax_p3.text(0.5, 0.5, "UMAP unavailable\n(install umap-learn)",
                       ha="center", va="center", transform=ax_p3.transAxes)

        # ── P4: Per-Step Entropy + Probability Curves ─────────────────────
        ax_p4.set_title("P4: Per-Step Entropy & Probability Curves",
                        fontsize=10, fontweight="bold")
        entropies = res["base_entropy"]    # (T,) values
        ax4b = ax_p4.twinx()

        ax_p4.plot(range(len(entropies)), entropies, "k-o",
                   label="Entropy H(a_τ)", linewidth=2)
        ax_p4.set_ylabel("Action Entropy (bits)", fontsize=9)
        ax_p4.set_xlabel("Optimisation step τ", fontsize=9)

        # Plot class probabilities across steps
        proba_steps = res["base_proba"]   # list of (1, 5) arrays
        for c in range(5):
            p_vals = [proba_steps[t][0, c] for t in range(len(proba_steps))]
            ax4b.plot(range(len(p_vals)), p_vals,
                      color=SUBTYPE_COLOURS[c], linestyle="--",
                      alpha=0.7, label=SUBTYPE_NAMES[c])

        ax4b.set_ylabel("P(subtype)", fontsize=9)
        ax4b.set_ylim([0, 1])
        ax_p4.legend(loc="upper left", fontsize=7)
        ax4b.legend(loc="upper right", fontsize=7)

        # ── P5: PAM50 Probability Shift ───────────────────────────────────
        ax_p5.set_title("P5: Baseline vs. Scenario PAM50 Shift",
                        fontsize=10, fontweight="bold")
        base_final  = res["base_proba"][-1][0]    # (5,) at z_T
        x_pos       = np.arange(5)
        width       = 0.35

        ax_p5.bar(x_pos - width/2, base_final, width, label="Baseline",
                  color=[SUBTYPE_COLOURS[c] for c in range(5)], alpha=0.7)

        if res["scen_proba"] is not None:
            scen_final = res["scen_proba"][-1][0]
            ax_p5.bar(x_pos + width/2, scen_final, width,
                      label=f"Scenario ({SCENARIO_NAMES.get(scenario_idx, '')})",
                      color=[SUBTYPE_COLOURS[c] for c in range(5)], alpha=0.4,
                      hatch="//")
            # Annotate delta
            for c in range(5):
                delta = scen_final[c] - base_final[c]
                if abs(delta) > 0.01:
                    ax_p5.annotate(f"Δ={delta:+.2f}",
                                   xy=(c + width/2, max(base_final[c], scen_final[c]) + 0.02),
                                   fontsize=7, ha="center")

        ax_p5.set_xticks(x_pos)
        ax_p5.set_xticklabels(list(SUBTYPE_NAMES.values()), fontsize=9)
        ax_p5.set_ylabel("P(subtype)", fontsize=9)
        ax_p5.set_ylim([0, 1.15])
        ax_p5.legend(fontsize=8)
        ax_p5.text(0.02, 0.98,
                   "Representation-space shifts only.\nNo causal interpretation.",
                   transform=ax_p5.transAxes, fontsize=7, va="top", color="gray")

        # ── P6: kNN Patient Retrieval ─────────────────────────────────────
        ax_p6.set_title("P6: kNN Latent Patient Retrieval (Cosine Similarity)",
                        fontsize=10, fontweight="bold")
        ax_p6.axis("off")

        if res["knn_indices"] is not None and self.patient_ids_bank is not None:
            rows = [["Patient ID", "Subtype", "Similarity", "Survival"]]
            for rank, (idx, sim) in enumerate(
                zip(res["knn_indices"], res["knn_distances"])
            ):
                pid   = self.patient_ids_bank[idx] if idx < len(self.patient_ids_bank) \
                        else f"PT-{idx:04d}"
                label = SUBTYPE_NAMES.get(
                    int(self.labels_bank[idx]) if self.labels_bank is not None else 0, "?"
                )
                surv  = f"{self.surv_bank[idx]:.0f} mo" \
                        if self.surv_bank is not None and idx < len(self.surv_bank) \
                        else "N/A"
                rows.append([pid, label, f"{sim:.3f}", surv])

            table = ax_p6.table(
                cellText=rows[1:],
                colLabels=rows[0],
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            ax_p6.text(0.5, -0.05,
                       "Research-level retrieval only. Not for clinical decision-making.",
                       transform=ax_p6.transAxes, fontsize=7, ha="center", color="gray")
        else:
            ax_p6.text(0.5, 0.5, "kNN bank not loaded.\nCall model.build_feedback_bank()",
                       ha="center", va="center", transform=ax_p6.transAxes, fontsize=9)

        # ── Figure Title ──────────────────────────────────────────────────
        fig.suptitle(
            f"CDLS Research Visualisation — Patient: {patient_id}  |  "
            f"⚠ Latent simulation tool. Not a clinical decision-support system.",
            fontsize=11, fontweight="bold", color="darkred"
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualisation → {save_path}")

        return fig


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CDLS Research Visualisation Interface")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to PPO best checkpoint (.pt)")
    parser.add_argument("--patient_id",  default="TCGA-3C-AAAU")
    parser.add_argument("--scenario",    type=int, default=None,
                        choices=[0, 1, 2], help="Scenario index (0=HER2, 1=LumA, 2=Hormone)")
    parser.add_argument("--bank_path",   default=None,
                        help="Path to z0 bank .pt file for kNN retrieval")
    parser.add_argument("--out",         default="results/visualisation.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CDLS(clinical_dim=20, dtwin=7, T=5)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    z_bank = labels_bank = patient_ids_bank = surv_bank = None
    if args.bank_path and os.path.exists(args.bank_path):
        bank_data = torch.load(args.bank_path)
        if isinstance(bank_data, dict):
            z_bank         = bank_data.get("z_bank")
            labels_bank    = bank_data.get("labels")
            patient_ids_bank = bank_data.get("patient_ids")
            surv_bank      = bank_data.get("surv_times")
        else:
            z_bank = bank_data
        if z_bank is not None:
            model.build_feedback_bank(z_bank)

    visualiser = CDLSVisualiser(
        model, device, z_bank, labels_bank, patient_ids_bank, surv_bank
    )

    # Create dummy batch for demo (replace with real patient data)
    dummy_batch = {
        "wsi":          torch.zeros(1, 10, 1024),
        "wsi_present":  torch.tensor([False]),
        "rna":          torch.randn(1, 20481),
        "rna_present":  torch.tensor([True]),
        "bcsc":         torch.zeros(1, 20, 7),
        "bcsc_present": torch.tensor([False]),
        "bcsc_len":     torch.tensor([0]),
        "clinical":     torch.randn(1, 20),
        "subtype":      torch.tensor([2]),
        "surv_time":    torch.tensor([40.2]),
        "event":        torch.tensor([0.0]),
    }

    result = visualiser.run_patient(dummy_batch, scenario_idx=args.scenario)
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    visualiser.plot_six_panels(result, patient_id=args.patient_id,
                               scenario_idx=args.scenario, save_path=args.out)
    print(f"Visualisation saved to {args.out}")


if __name__ == "__main__":
    main()
