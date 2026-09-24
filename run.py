from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
# The bundled local dependencies were built for CPython 3.12.  Do not put
# them ahead of a user's compatible site-packages on another Python version.
if VENDOR.exists() and sys.version_info[:2] == (3, 12):
    sys.path.insert(0, str(VENDOR))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import numpy as np

from src.analysis import register, robustness, uncertainty
from src.io_utils import load_ply, write_ply
from src.metrics import bidirectional_metrics, directional_metrics, subset_metrics
from src.preprocess import estimate_spacing
from src.registration import apply
from src.visualization import distance_map, overlay


def rounded(value):
    if isinstance(value, np.ndarray):
        return [rounded(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def inspect_line(label: str, info: dict) -> str:
    return (f"{label}: vertices={info['vertices_valid']}, faces={info['faces']}, "
            f"colors={info['has_colors']}, normals={info['has_normals']}, "
            f"extent={np.round(info['extent'], 3).tolist()}, centroid={np.round(info['centroid'], 3).tolist()}")


def main() -> None:
    overall = time.perf_counter()
    output = ROOT / "resultados"
    output.mkdir(exist_ok=True)
    for name in (
        "antes.png", "depois.png", "mapa_distancia.png", "malha2_alinhada.ply",
        "mapa_distancia_vertices.ply", "metricas.json", "metrics.json", "relatorio.txt",
        "transformacao_final.json",
    ):
        (output / name).unlink(missing_ok=True)
    reference_path, moving_path = ROOT / "malha 1.ply", ROOT / "malha 2.ply"
    if not reference_path.exists() or not moving_path.exists():
        raise FileNotFoundError("Esperados: malha 1.ply e malha 2.ply na pasta do projeto.")

    loading_started = time.perf_counter()
    reference, reference_info = load_ply(reference_path)
    moving, moving_info = load_ply(moving_path)
    loading_seconds = time.perf_counter() - loading_started
    print("\nINSPECAO DOS ARQUIVOS")
    print(inspect_line("Referencia (malha 1.ply)", reference_info))
    print(inspect_line("Movel (malha 2.ply)", moving_info))
    print("Interpretacao: PLYs sem elemento face; nuvens de pontos com RGB e normais.")

    preprocessing_started = time.perf_counter()
    spacing = estimate_spacing(reference.points)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    unit_note = "unidade nativa do PLY; unidade fisica nao declarada"
    final_transform, registration_info, stable_mask = register(reference.points, moving.points, reference.normals, spacing)
    aligned = moving.transformed(final_transform)
    coverage_threshold = spacing * 8
    before_m2r_distances, before_m2r = directional_metrics(moving.points, reference.points, coverage_threshold, "moving -> reference")
    before_r2m_distances, before_r2m = directional_metrics(reference.points, moving.points, coverage_threshold, "reference -> moving")
    after_m2r_distances, after_m2r = directional_metrics(aligned.points, reference.points, coverage_threshold, "moving -> reference")
    after_r2m_distances, after_r2m = directional_metrics(reference.points, aligned.points, coverage_threshold, "reference -> moving")
    before_bidirectional = bidirectional_metrics(before_m2r, before_r2m)
    after_bidirectional = bidirectional_metrics(after_m2r, after_r2m)
    candidate_threshold = float(np.quantile(after_m2r_distances, 0.90))
    candidate_mask = after_m2r_distances > candidate_threshold
    candidate_metrics = subset_metrics(
        after_m2r_distances[candidate_mask],
        "moving vertices with post-registration moving -> reference residual strictly above P90; heuristic candidate discrepancy region",
    )
    candidate_metrics.update({
        "percentile": 0.90, "threshold_native": candidate_threshold,
        "approximately_percent_of_moving_points": float(100 * candidate_mask.mean()),
        "spatially_contiguous": False,
    })

    visualization_started = time.perf_counter()
    overlay(reference.points, moving.points, output / "antes.png", "Antes do alinhamento")
    overlay(reference.points, aligned.points, output / "depois.png", "Depois do alinhamento")
    color_clip = float(np.quantile(after_m2r_distances, .95))
    colors = distance_map(aligned.points, after_m2r_distances, output / "mapa_distancia.png", color_clip)
    visualization_seconds = time.perf_counter() - visualization_started
    write_ply(output / "malha2_alinhada.ply", aligned)
    write_ply(output / "mapa_distancia_vertices.ply", aligned, colors)

    uncertainty_info = uncertainty(reference.points, moving.points, reference.normals, spacing)
    robustness_info = robustness(reference.points, moving.points, reference.normals, spacing, final_transform)
    timings = {
        "loading_seconds": loading_seconds, "preprocessing_seconds": preprocessing_seconds,
        "coarse_and_icp_seconds": registration_info["registration_seconds"],
        "visualization_seconds": visualization_seconds, "total_seconds": time.perf_counter() - overall,
    }
    transform_payload = {
        "convention": "p_reference = T @ p_moving (homogeneous 4x4, column coordinates)",
        "reference": reference_path.name, "moving": moving_path.name,
        "matrix": final_transform.tolist(), "rotation_3x3": final_transform[:3, :3].tolist(),
        "translation_native_units": final_transform[:3, 3].tolist(),
        "rotation_determinant": float(np.linalg.det(final_transform[:3, :3])),
        "rotation_orthonormality_error": float(np.linalg.norm(final_transform[:3, :3].T @ final_transform[:3, :3] - np.eye(3))),
        "translation_unit": "native PLY unit; physical unit undeclared",
    }
    metrics = {
        "unit": {"reporting_unit": "native PLY unit", "physical_unit_status": "not declared by input PLY"},
        "data_interpretation": "point clouds; no faces present in either PLY",
        "spacing_estimate_native": spacing, "inspection": {"reference": reference_info, "moving": moving_info},
        "coverage_threshold_native": coverage_threshold,
        "before": {"moving_to_reference": before_m2r, "reference_to_moving": before_r2m, "bidirectional": before_bidirectional},
        "after": {"moving_to_reference": after_m2r, "reference_to_moving": after_r2m, "bidirectional": after_bidirectional},
        "stable_region": {**registration_info["stable_region"], "source": "moving points", "selection_stage": "after coarse PCA, before ICP", "used_for_icp": True, "heuristic": "nearest-neighbor residual consensus; not anatomical or clinical"},
        "candidate_discrepancy_region": candidate_metrics,
        "icp": {"method": "multi-scale point-to-plane ICP", "robust_kernel": "Cauchy", "stable_mask_used": True, "stages": registration_info["icp_stages"]},
        "registration": registration_info, "uncertainty": uncertainty_info, "robustness": robustness_info, "timings": timings,
    }
    (output / "transformacao_final.json").write_text(json.dumps(rounded(transform_payload), indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(rounded(metrics), indent=2), encoding="utf-8")
    report = f"""HACKATHON ALLIAGE - DESAFIO 04\nREGISTRO RIGIDO DE CAPTURAS 3D\n\nReferencia: {reference_path.name}\nMovel: {moving_path.name}\nUnidade: {unit_note}\n\nDados: nuvens de pontos PLY sem faces. Todas as estatisticas direcionais usam NN Euclidiano e todos os pontos da fonte.\n\nRMS antes (M->R): {before_m2r['rms_native']:.6f}\nRMS depois (M->R): {after_m2r['rms_native']:.6f}\nRMS depois (R->M): {after_r2m['rms_native']:.6f}\nRMS bidirecional depois: {after_bidirectional['rms_bidirectional_native']:.6f}\nMediana depois (M->R): {after_m2r['median_native']:.6f}\nP95 depois (M->R): {after_m2r['p95_native']:.6f}\nP99 depois (M->R): {after_m2r['p99_native']:.6f}\nMaximo NN direcionado depois (M->R): {after_m2r['maximum_directed_nearest_neighbor_distance_native']:.6f}\nCobertura <= {coverage_threshold:.6f} unidade nativa: {after_m2r['coverage_within_threshold_percent']:.2f}%\n\nRegiao estavel: {registration_info['stable_region']['selected_points']} pontos ({registration_info['stable_region']['fraction']:.0%}), heuristica apos PCA e usada pelo ICP.\nRegiao candidata a alteracao: {candidate_metrics['points_evaluated']} pontos acima de P90 ({candidate_threshold:.6f}); heuristica nao anatomica.\n\nVariabilidade RMS (bootstrap): media {uncertainty_info['rms_mean_native']:.6f}; sd {uncertainty_info['rms_std_native']:.6f}.\nTempo total: {timings['total_seconds']:.3f} s\n"""
    (output / "relatorio.txt").write_text(report, encoding="utf-8")

    print("\n" + "=" * 40)
    print("HACKATHON ALLIAGE - DESAFIO 04")
    print("REGISTRO RIGIDO DE CAPTURAS 3D")
    print("=" * 40)
    print(f"REFERENCIA: {reference_path.name}\nMOVEL: {moving_path.name}\nUNIDADE: unidade nativa do PLY\nSTATUS: unidade fisica nao declarada")
    print(f"VERTICES: referencia {len(reference.points)} | movel {len(moving.points)}")
    print(f"FACES: referencia {reference_info['faces']} | movel {moving_info['faces']}")
    print(f"ALINHAMENTO GROSSEIRO: {registration_info['coarse']['method']}")
    print("ICP: point-to-plane, tres escalas, kernel Cauchy e mascara estavel")
    print(f"RMS ANTES (M->R): {before_m2r['rms_native']:.6f}\nRMS DEPOIS (M->R): {after_m2r['rms_native']:.6f}")
    print(f"RMS ANTES (R->M): {before_r2m['rms_native']:.6f}\nRMS DEPOIS (R->M): {after_r2m['rms_native']:.6f}")
    print(f"RMS BIDIRECIONAL DEPOIS: {after_bidirectional['rms_bidirectional_native']:.6f}")
    print(f"MEDIANA (M->R): {after_m2r['median_native']:.6f}\nP95 (M->R): {after_m2r['p95_native']:.6f}\nP99 (M->R): {after_m2r['p99_native']:.6f}\nMAXIMO NN DIRECIONADO (M->R): {after_m2r['maximum_directed_nearest_neighbor_distance_native']:.6f}")
    print(f"COBERTURA <= {coverage_threshold:.6f} unidade nativa: {after_m2r['coverage_within_threshold_percent']:.2f}%")
    print(f"REGIAO ESTAVEL: {registration_info['stable_region']['selected_points']} pontos\nREGIAO CANDIDATA A ALTERACAO: {candidate_metrics['points_evaluated']} pontos (d > P90)")
    print(f"VARIABILIDADE RMS (bootstrap): media = {uncertainty_info['rms_mean_native']:.6f}; sd = {uncertainty_info['rms_std_native']:.6f}")
    print(f"TEMPO TOTAL: {timings['total_seconds']:.3f} s")
    print(f"TRANSFORMACAO: {output / 'transformacao_final.json'}")
    print(f"MALHA ALINHADA: {output / 'malha2_alinhada.ply'}")
    print(f"METRICAS: {output / 'metrics.json'}")
    print(f"MAPA DE DISTANCIA POINT-WISE: {output / 'mapa_distancia.png'}")
    print("=" * 40)


if __name__ == "__main__":
    main()
