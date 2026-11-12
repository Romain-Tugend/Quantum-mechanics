#!/usr/bin/env python3
import concurrent.futures
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parent
JOB = REPO / "job_mpi.sh"
LOG_DIR = REPO / "logs"
MAKEFILE = "makefile"

MAX_PROCESSORS = 32
BASE_NX = 4096
BASE_NY = 256
REPEATS = 3
TIME_LIMIT = "00:05:00"

TYPE = "weak" if (len(sys.argv) > 1 and sys.argv[1] == "w") else "strong"


def run(cmd, check=True):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=REPO, check=check)


def wall_times(path):
    if not path.exists():
        return []
    return [
        float(t)
        for t in re.findall(r"wall_time(?:_\d+)?=([0-9.]+)", path.read_text())
    ]


def run_single_job(n):
    if TYPE == "strong":
        # Grille fixe 8192x256 répartie sur n cœurs (64 lignes/cœur à N=32)
        nx, ny = BASE_NX * 4, BASE_NY 
    else:
        # Charge par cœur constante (4096x256 à N=1 jusqu'à ~33 500 000 à N=32)
        scale = n
        nx = int(round(BASE_NX * scale))
        ny = int(round(BASE_NY ))

    out = LOG_DIR / f"{TYPE}_N{n}.out"
    err = LOG_DIR / f"{TYPE}_N{n}.err"

    run(
        [
            "sbatch",
            "--wait",
            f"--job-name=Schrod_{TYPE}_N{n}",
            f"--ntasks={n}",
            f"--time={TIME_LIMIT}",
            f"--output={out}",
            f"--error={err}",
            str(JOB),
            str(REPEATS),
            str(nx),
            str(ny),
        ],
        check=False,
    )

    times = wall_times(out)
    if not times:
        return n, nx, ny, 0.0, 0.0, False

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return n, nx, ny, mean, std, True


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    run(["make", "-f", MAKEFILE, "clean"])
    run(["make", "-f", MAKEFILE])

    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_PROCESSORS
    ) as executor:
        futures = [
            executor.submit(run_single_job, n)
            for n in range(1, MAX_PROCESSORS + 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            n, nx, ny, mean, std, success = future.result()
            if success:
                results.append((n, nx, ny, mean, std))
                print(
                    f"[OK] N={n:2d} ({nx}x{ny}) : {mean:.4f}s ± {std:.4f}s",
                    flush=True,
                )
            else:
                print(
                    f"[ÉCHEC] N={n:2d} ({nx}x{ny}) — voir logs/{TYPE}_N{n}.err",
                    flush=True,
                )

    results.sort(key=lambda x: x[0])
    t1 = results[0][3] if (results and results[0][0] == 1) else None

    output_csv = REPO / f"{TYPE}_results.csv"
    with open(output_csv, "w") as file:
        file.write("n,Nx,Ny,mean,std,speedup,efficiency\n")
        for n, nx, ny, mean, std in results:
            if t1 and mean > 0:
                speedup = t1 / mean
                eff = speedup / n if TYPE == "strong" else t1 / mean
            else:
                speedup, eff = float("nan"), float("nan")
            file.write(
                f"{n},{nx},{ny},{mean:.6f},{std:.6f},{speedup:.4f},{eff:.4f}\n"
            )

    print(f"Terminé. Résultats sauvegardés dans {output_csv}")
