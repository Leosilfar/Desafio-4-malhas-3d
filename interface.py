"""Interface simples para registrar duas nuvens de pontos PLY."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "resultados"


class RegistrationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Alinhamento 3D | Análise de nuvens")
        self.geometry("850x690")
        self.minsize(700, 560)
        self.reference = tk.StringVar()
        self.moving = tk.StringVar()
        self.status = tk.StringVar(value="Selecione os dois arquivos PLY para começar.")
        self.events: queue.Queue = queue.Queue()
        self.preview = None
        self._build()
        self.after(150, self._poll)

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Análise de alinhamento 3D", font=("TkDefaultFont", 20, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Envie duas nuvens de pontos PLY para calcular o registro e as diferenças.",
                  wraplength=760).pack(anchor="w", pady=(4, 18))
        self._file_row(frame, "Nuvem de referência", self.reference)
        self._file_row(frame, "Nuvem móvel", self.moving)
        self.run_button = ttk.Button(frame, text="Executar análise", command=self._run)
        self.run_button.pack(anchor="w", pady=(2, 14))
        ttk.Label(frame, textvariable=self.status).pack(anchor="w", pady=(0, 8))
        self.metrics = ttk.LabelFrame(frame, text="Resumo", padding=10)
        self.metrics.pack(fill="x", pady=(0, 12))
        self.summary = ttk.Label(self.metrics, text="Os resultados aparecerão aqui.", justify="left")
        self.summary.pack(anchor="w")
        self.image = ttk.Label(frame, anchor="center")
        self.image.pack(fill="both", expand=True)

    def _file_row(self, parent, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text=label, width=23).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row, text="Escolher PLY…", command=lambda: self._choose(variable)).pack(side="right")

    def _choose(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(title="Selecione uma nuvem PLY", filetypes=[("Nuvem PLY", "*.ply")])
        if path:
            variable.set(path)

    def _run(self) -> None:
        reference, moving = self.reference.get().strip(), self.moving.get().strip()
        if not reference or not moving:
            messagebox.showinfo("Arquivos necessários", "Selecione a nuvem de referência e a nuvem móvel.")
            return
        if not Path(reference).is_file() or not Path(moving).is_file():
            messagebox.showerror("Arquivo não encontrado", "Confira os caminhos dos arquivos selecionados.")
            return
        self.run_button.configure(state="disabled")
        self.status.set("Análise em andamento. Isso pode levar alguns minutos…")
        self.summary.configure(text="")
        self.image.configure(image="")
        project_python = ROOT / ".venv" / "bin" / "python"
        interpreter = str(project_python) if project_python.is_file() else sys.executable
        command = [interpreter, str(ROOT / "run.py"), "--referencia", reference,
                   "--movel", moving, "--saida", str(OUTPUT)]
        threading.Thread(target=self._worker, args=(command,), daemon=True).start()

    def _worker(self, command: list[str]) -> None:
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.events.put((result.returncode, result.stdout, result.stderr))
        except Exception as exc:
            self.events.put((-1, "", str(exc)))

    def _poll(self) -> None:
        try:
            code, stdout, stderr = self.events.get_nowait()
        except queue.Empty:
            self.after(150, self._poll)
            return
        self.run_button.configure(state="normal")
        if code:
            self.status.set("A análise não foi concluída.")
            messagebox.showerror("Erro na análise", stderr.strip() or stdout.strip() or "Falha ao executar o registro.")
        else:
            self.status.set(f"Concluído. Arquivos salvos em: {OUTPUT}")
            self._show_results()
        self.after(150, self._poll)

    def _show_results(self) -> None:
        try:
            data = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
            after = data["after"]
            m2r = after["moving_to_reference"]
            lines = [
                f"RMS bidirecional: {after['bidirectional']['rms_bidirectional_native']:.6f} (unidade nativa)",
                f"RMS móvel → referência: {m2r['rms_native']:.6f}    |    Mediana: {m2r['median_native']:.6f}",
                f"P95: {m2r['p95_native']:.6f}    |    Cobertura: {m2r['coverage_within_threshold_percent']:.2f}%",
                "A unidade física não é declarada nos arquivos PLY.",
            ]
            self.summary.configure(text="\n".join(lines))
            path = OUTPUT / "mapa_distancia.png"
            self.preview = tk.PhotoImage(file=str(path))
            scale = max(1, (self.preview.width() + 699) // 700, (self.preview.height() + 349) // 350)
            self.preview = self.preview.subsample(scale, scale)
            self.image.configure(image=self.preview)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self.summary.configure(text=f"Análise concluída. Consulte os resultados em {OUTPUT}\n{exc}")


if __name__ == "__main__":
    RegistrationApp().mainloop()
